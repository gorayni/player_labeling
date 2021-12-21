from argparse import ArgumentParser
from pathlib import Path


def get_processed_filepath(match_path, args):
    if args.file_type == 'velocity':
        return match_path.joinpath('velocity_results.npy')
    elif args.file_type == 'preliminary_player_labels':
        return match_path.joinpath('preliminary_player_labels.npy')
    elif args.file_type == 'train':
        return match_path.joinpath('player_labeling', 'weights', args.model_name, 'initial_model.pth.tar')
    elif args.file_type == 'segmentation':
        return match_path.joinpath(f'segmentation_results_{args.half+1}_HQ.npy')


def main(args):
    filepaths_to_processed = []
    for match_path in args.dataset_path.glob('*/*/*/'):
        if not match_path.is_dir():
            continue

        if args.file_type != 'segmentation':
            processed_fpath = get_processed_filepath(match_path, args)
            if not processed_fpath.exists():
                filepaths_to_processed.append(match_path)
        else:
            for i in range(2):
                video_path = match_path.joinpath(f'{i + 1}_HQ.mkv')
                args.half = i
                processed_fpath = get_processed_filepath(match_path, args)
                if not processed_fpath.exists():
                    filepaths_to_processed.append(video_path)
    filepaths_to_processed = sorted(filepaths_to_processed)
    for f in filepaths_to_processed:
        print(f)


if __name__ == '__main__':
    parser = ArgumentParser(description='Split files to be processed')
    parser.add_argument('-f', '--file_type',
                        help='Type of the files to be processed',
                        type=str)
    parser.add_argument('-m', '--model_name', required=False,
                        help='Model name for the CNN to be trained (default: mobilenet_v3_large)',
                        default="mobilenet_v3_large", type=str)
    parser.add_argument('-d', '--dataset_path', required=False,
                        help='Path for SoccerNet dataset (default: data/soccernet)',
                        default="data/soccernet", type=lambda p: Path(p))
    main(parser.parse_args())
