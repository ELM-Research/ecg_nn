import torch
from tqdm import tqdm
import numpy as np

from utils.gpu_setup import is_main
from utils.viz import plot_forecast
from utils.eval_stats import forecast_metrics
from utils.runner_helpers import batch_to_device
from utils.dir_file import DirFileManager
from configs.constants import PTB_ORDER

def eval_forecasting(nn, dataloader, args):
    if getattr(args, "signal_head", False):
        return _eval_forecasting_signal_head(nn, dataloader, args)
    return _eval_forecasting_tokens(nn, dataloader, args)


def _eval_forecasting_signal_head(nn, dataloader, args):
    show_progress = is_main()
    nn.eval()
    progress = tqdm(dataloader, desc="Evaluating Forecasting (Signal Head)", disable=not show_progress, leave=False)
    device = next(nn.parameters()).device
    data_repr = dataloader.dataset.data_representation
    condition_name = f"{args.condition}" if args.condition else f"{args.condition}_{args.condition_lead}"
    data_names = "_".join(args.data)
    plot_dir = f"{args.run_dir}/{data_names}_{args.forecast_ratio}_{args.bpe_symbolic_len}_{condition_name}_signal_head"
    DirFileManager.ensure_directory_exists(folder=plot_dir)

    if args.condition:
        n_leads = 1
        lead_names = [PTB_ORDER[args.condition_lead]]
    else:
        n_leads = len(PTB_ORDER)
        lead_names = PTB_ORDER

    all_sig = []
    max_seq_len = args.bpe_symbolic_len
    signal_shape = (1, n_leads, args.segment_len)
    num_steps = getattr(args, "signal_head_num_steps", 50)
    forecast_start = int(args.segment_len * (1 - args.forecast_ratio))

    with torch.no_grad():
        for step, batch in enumerate(progress):
            report = batch["report"][0]
            gt_signal = batch["signal"].numpy()
            mn, mx = batch["min"][0].item(), batch["max"][0].item()
            batch = {k: batch_to_device(v, device) for k, v in batch.items()}
            tgt_ids = batch["tgt_ids"]
            if tgt_ids.size(1) >= max_seq_len:
                tgt_ids = tgt_ids[:, -(max_seq_len - 1):]
            labels = batch.get("labels")
            max_new_tokens = labels.shape[1] if labels is not None else max_seq_len - tgt_ids.size(1)
            pred_signal = nn.generate_signal(tgt_ids, max_new_tokens, signal_shape, num_steps).cpu().numpy()
            gt_denorm = data_repr.denormalize(gt_signal[0].ravel(), mn, mx)
            pred_denorm = data_repr.denormalize(pred_signal[0].ravel(), mn, mx)
            full_gt = gt_denorm[:n_leads * args.segment_len].reshape(n_leads, args.segment_len)
            full_pred = pred_denorm[:n_leads * args.segment_len].reshape(n_leads, args.segment_len)
            all_sig.append(forecast_metrics(full_pred[:, forecast_start:].ravel(), full_gt[:, forecast_start:].ravel()))
            if step < 20:
                plot_forecast(full_gt, full_pred, 0, n_leads * args.segment_len, n_leads * args.segment_len,
                              report, f"{plot_dir}/plot_{step}.png",
                              segment_len=args.segment_len, leads=lead_names, sf=args.sf, ctx_per_lead=forecast_start)
            if step > 3:
                break

    metrics = {}
    for k in all_sig[0]:
        metrics[k] = float(np.nanmean([s[k] for s in all_sig]))
    print("Forecast (Signal Head) | " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def _eval_forecasting_tokens(nn, dataloader, args):
    show_progress = is_main()
    nn.eval()
    progress = tqdm(
        dataloader,
        desc="Evaluating Forecasting",
        disable=not show_progress,
        leave=False,
    )
    device = next(nn.parameters()).device
    data_repr = dataloader.dataset.data_representation
    condition_name = f"{args.condition}" if args.condition else f"{args.condition}_{args.condition_lead}"
    data_names = "_".join(args.data)
    plot_dir = f"{args.run_dir}/{data_names}_{args.forecast_ratio}_{args.bpe_symbolic_len}_{condition_name}_{args.lead_tokens}"
    DirFileManager.ensure_directory_exists(folder=plot_dir)
    all_acc, all_sig = [], []
    max_seq_len = args.bpe_symbolic_len

    if args.condition:
        n_leads = len(args.condition_lead)
        lead_names = [PTB_ORDER[i] for i in args.condition_lead]
    else:
        n_leads = len(PTB_ORDER)
        lead_names = PTB_ORDER
    n_total = n_leads * args.segment_len

    with torch.no_grad():
        for step, batch in enumerate(progress):
            report = batch["report"][0]
            labels = batch["labels"].numpy()
            batch = {k: batch_to_device(v, device) for k, v in batch.items()}
            tgt_ids = batch["tgt_ids"]
            if tgt_ids.size(1) >= max_seq_len:
                tgt_ids = tgt_ids[:, -(max_seq_len - 1):]
            pred = nn.generate(tgt_ids, max_new_tokens=labels.shape[1],
                               return_new_only=True, return_logits=False).cpu().numpy()
            gt = labels[:, :pred.shape[1]]
            all_acc.append((pred == gt).mean())
            for i in range(len(gt)):
                mn, mx = batch["min"][i].item(), batch["max"][i].item()
                ps = data_repr.decode(pred[i].tolist(), mn, mx)
                gs = data_repr.decode(gt[i].tolist(), mn, mx)
                all_sig.append(forecast_metrics(ps, gs))
                if step < 20 and i == 0:
                    ctx = data_repr.decode(tgt_ids[0].tolist(), mn, mx)
                    n_ctx = len(ctx)
                    n_gt_end = min(len(ctx) + len(gs), n_total)
                    n_pred_end = min(len(ctx) + len(ps), n_total)
                    full_gt = np.pad(np.concatenate([ctx, gs]), (0, max(0, n_total - len(ctx) - len(gs))))[:n_total].reshape(n_leads, args.segment_len)
                    full_pred = np.pad(np.concatenate([ctx, ps]), (0, max(0, n_total - len(ctx) - len(ps))))[:n_total].reshape(n_leads, args.segment_len)
                    plot_forecast(full_gt, full_pred, n_ctx, n_gt_end, n_pred_end,
                                  report, f"{plot_dir}/plot_{step}.png",
                                  segment_len=args.segment_len, leads=lead_names, sf=args.sf)
            if step > 3:
                break

    metrics = {"accuracy": float(np.nanmean(all_acc))}
    for k in all_sig[0]:
        metrics[k] = float(np.nanmean([s[k] for s in all_sig]))
    print("Forecast | " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics
