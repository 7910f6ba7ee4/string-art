import argparse
from datetime import datetime

from art import ComputeStringArt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Full input file path.')
    parser.add_argument('-os', '--output_summary', type=str, required=False,
                        help='Full output file path. Defaults to output/string_art_summary_{timestamp}')
    parser.add_argument('-o', '--output', type=str, required=False,
                        help='Full output file path. Defaults to output/string_art_{timestamp}')
    parser.add_argument('-s', '--spokes', type=int, required=False, default=360,
                        help='Number of spokes. Defaults to 360.')
    parser.add_argument('-r', '--res', type=int, required=False, default=180,
                        help='Resolution of radon transform. Defaults to 180.')
    parser.add_argument('-l', '--lines', type=int, required=False, default=400,
                        help='Amount of lines to draw. Defaults to 400.')

    args = parser.parse_args()
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.output_summary is None:
        args.output_summary = f"output/string_art_summary_{now}"
    if args.output is None:
        args.output = f"output/string_art_{now}"

    string_art = ComputeStringArt(
        args.input,
        n_spokes=args.spokes,
        radon_res=args.res,
        repetitions=args.lines)
    string_art.render()
    string_art.display_art(args.output)
    string_art.display_summary(args.output_summary)
