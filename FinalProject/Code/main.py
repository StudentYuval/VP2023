from collections import OrderedDict
import time
import json
from funcs_and_structs import ProjectPaths
from stabilization import stabilization_block
from matting import matting_block
from tracking import tracking_block
from background_subtraction import bg_subtraction_block
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--show-work', action='store_true', help='displays visualizations of the work done by the algorithm', dest='show_work')
args = parser.parse_args()

# student id numbers
ID1 = '206299463'
ID2 = '312497084'
paths = ProjectPaths(suffix=f'{ID1}_{ID2}')

start = time.time()
timing_json = OrderedDict()
tracking_json = OrderedDict()
timing_json["time_to_stabilize"] = stabilization_block(paths.Input, paths.Stabilized, show_work=args.show_work)
timing_json["time_to_binary"] = bg_subtraction_block(paths.Stabilized, paths.Extracted, paths.Binary, show_work=args.show_work)
timing_json["time_to_alpha"], timing_json["time_to_matted"] = matting_block(paths.Extracted, paths.Binary, paths.Background, paths.Matted, paths.Alpha)
timing_json["time_to_output"] = tracking_block(paths.Matted, paths.Output, tracking_json)

with open(paths.Timing, 'w') as f:
    json.dump(timing_json, f, indent=4)

with open(paths.Tracking, 'w') as f:
    json.dump(tracking_json, f, indent=4)

total_time = time.time() - start
print(f'Finished processing, total duration: {total_time//60:.0f}m {total_time%60:.0f}s')
