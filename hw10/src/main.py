import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings("ignore")

import glob
import os
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--myId", help="Your student id", default="111598033", type=str)

    parser.add_argument(
        "--targetId",
        help="Target student id",
        type=str,
    )

    args = parser.parse_args()
    return args


def compareWithTarget(
    args: ArgumentParser,
    targetID: str,
    word_list: list,
    markDatabase: dict,
    loss_fn,
    device,
) -> None:
    success = totalLPIPS = totalMSE = totalSSIM = 0
    pbar = tqdm(word_list, desc="Calculating similarity")
    for i, word_path in enumerate(pbar):
        try:
            _, filename = os.path.split(word_path)

            # Check file exist
            if not (
                os.path.exists(f"./1_138_{args.myId}/{filename}")
                and os.path.exists(f"./1_138_{targetID}/{filename}")
            ):
                raise FileNotFoundError

            # 自己的手寫字圖片路徑
            MyWord_ = cv2.imread(
                f"./1_138_{args.myId}/{filename}", cv2.IMREAD_GRAYSCALE
            )
            MyWord = torch.from_numpy(MyWord_).unsqueeze(0).unsqueeze(0).float() / 255.0
            MyWord = MyWord.to(device)

            # 別人的手寫字圖片路徑
            TargetWord_ = cv2.imread(
                f"./1_138_{targetID}/{filename}", cv2.IMREAD_GRAYSCALE
            )
            TargetWord = (
                torch.from_numpy(TargetWord_).unsqueeze(0).unsqueeze(0).float() / 255.0
            )
            TargetWord = TargetWord.to(device)

            # 計算分數
            mse = np.mean((MyWord_ - TargetWord_) ** 2)  # 計算MSE
            ssim_score = ssim(MyWord_, TargetWord_, win_size=7)  # 計算SSIM相似度
            lpips_distance = loss_fn(MyWord, TargetWord)  # 計算LPIPS距離

            # Accumulate the scores
            success += 1
            totalMSE += mse
            totalSSIM += ssim_score
            totalLPIPS += lpips_distance.item()
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error occured at {word_path}, {e}")
            exit()

    if success != 0:
        if targetID not in markDatabase:
            markDatabase[targetID] = dict()
        markDatabase[targetID]["MSE"] = totalMSE / success
        markDatabase[targetID]["SSIM"] = totalSSIM / success
        markDatabase[targetID]["LPIPS"] = totalLPIPS / success
        print(
            f'Target: {targetID} \
            MSE: {markDatabase[targetID]["MSE"]:.5f} \
            SSIM: {markDatabase[targetID]["SSIM"]:.5f} \
            LPIPS: {markDatabase[targetID]["LPIPS"]:.5f}'
        )


def printMostSimilar(args: ArgumentParser, markDatabase: dict) -> None:
    """
    Print the most similar comparison result
    """
    minMSE_ID = min(markDatabase, key=lambda x: markDatabase[x]["MSE"])
    maxSSIM_ID = max(markDatabase, key=lambda x: markDatabase[x]["SSIM"])
    minLPIPS_ID = min(markDatabase, key=lambda x: markDatabase[x]["LPIPS"])

    print(f"Compare ID: {args.myId}")
    print(f'Most similar by MSE: {minMSE_ID} {markDatabase[minMSE_ID]["MSE"]:.5f}')
    print(f'Most similar by SSIM: {maxSSIM_ID} {markDatabase[maxSSIM_ID]["SSIM"]:.5f}')
    print(
        f'Most similar by LPIPS: {minLPIPS_ID} {markDatabase[minLPIPS_ID]["LPIPS"]:.5f}'
    )


def main(args):
    word_list = glob.glob(f"1_138_{args.myId}/*.png")
    markDatabase = dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入預訓練的LPIPS模型
    loss_fn = lpips.LPIPS(net="alex").to(device)

    compareWithTarget(args, args.targetId, word_list, markDatabase, loss_fn, device)

    # 輸出成績並保存結果
    printMostSimilar(args, markDatabase)


if __name__ == "__main__":
    args = parse_args()
    main(args)
