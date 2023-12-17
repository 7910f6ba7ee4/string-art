# String Art Testing
An attempt at creating computationally efficient string art using fourier transforms, more specifically a radon transform.
Does not work well currently, struggles with complex images, although can create a rough shape of a simple image
such as a rectangle.

### Installation
First clone and enter the repository.

```
git clone https://github.com/7910f6ba7ee4/string-art.git
cd string-art
```
It is recommended to put input images in the `input/` folder, but the script does not automatically
pull from there.


### Usage
Use `python3 main.py` with the following options.
```
options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Full input file path.
  -os OUTPUT_SUMMARY, --output_summary OUTPUT_SUMMARY
                        Full output file path. Defaults to output/string_art_summary_{timestamp}
  -o OUTPUT, --output OUTPUT
                        Full output file path. Defaults to output/string_art_{timestamp}
  -s SPOKES, --spokes SPOKES
                        Number of spokes. Defaults to 360.
  -r RES, --res RES     Resolution of radon transform. Defaults to 180.
  -l LINES, --lines LINES
                        Amount of lines to draw. Defaults to 400.
```
Access this menu with `python3 main.py --help`.