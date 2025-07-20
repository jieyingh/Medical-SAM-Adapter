import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from skimage import measure
from function import transform_prompt
import cfg

def keep_largest_connected_component(mask):
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc.astype(np.uint8)

def save_predictions(args, loader, model):
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Saving predictions"):
            images = batch['image'].to(dtype=torch.float32, device=f'cuda:{args.gpu_device}')
            filenames = batch['image_meta_dict']['filename_or_obj']
            preds = model(images)
            probs = torch.sigmoid(preds)
            preds_bin = (probs > 0.5).float()

            for i in range(images.size(0)):
                pred_mask = preds_bin[i, 0].cpu().numpy().astype(np.uint8)
                if args.keep_largest:
                    pred_mask = keep_largest_connected_component(pred_mask)

                out_path = os.path.join(args.output_dir, f"{filenames[i]}.png")
                Image.fromarray(pred_mask * 255).save(out_path)  # save as 8-bit image

def evaluate_and_save_predictions(args, dataloader, net, output_dir):
    args = cfg.parse_args()
    device = torch.device('cuda:' + str(args.gpu_device) if args.gpu else 'cpu')
    net.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Saving predictions")):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            names = batch['image_meta_dict']['filename_or_obj']

            if 'pt' in batch:
                pt = batch['pt']
                point_labels = batch['p_label']
                if len(point_labels.shape) == 1:
                    point_labels = point_labels.unsqueeze(1)
                    pt = pt.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
                if args.net == 'efficient_sam':
                    _, h, w = imgs.shape[-3:]
                    coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                sparse_embeddings, dense_embeddings = net.prompt_encoder(
                    points=(coords_torch, labels_torch),
                    boxes=None,
                    masks=None
                )
            else:
                sparse_embeddings, dense_embeddings = None, None

            imgs_input = net.preprocess(imgs)
            image_embeddings = net.image_encoder(imgs_input)
            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            elif args.net == 'efficient_sam':
                sparse_embeddings = sparse_embeddings.view(
                    sparse_embeddings.shape[0], 1,
                    sparse_embeddings.shape[1], sparse_embeddings.shape[2]
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    multimask_output=False,
                )

            pred = torch.nn.functional.interpolate(
                pred,
                size=(args.out_size, args.out_size),
                mode='bilinear',
                align_corners=False
            )
            pred_bin = (pred > 0.5).float().squeeze(1).cpu().numpy().astype(np.uint8)

            for idx in range(len(names)):
                img_name = os.path.basename(names[idx])
                save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_{args.label}.png")
                mask_np = pred_bin[idx] * 255  # 0 and 255 for 8-bit
                Image.fromarray(mask_np).save(save_path)