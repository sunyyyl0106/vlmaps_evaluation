import sys
import os
sys.path.append("..")

from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat

import numpy as np


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    vlmap = VLMap(config.map_config)
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    #vlmap.create_map(data_dirs[config.scene_id])
    vlmap.load_map(data_dirs[config.scene_id])

    result_dir = os.path.join(data_dirs[config.scene_id], 'heldout_result.txt')

    map_accuracy, map_miou, lseg_accuracy, lseg_miou = vlmap.evaluate_heldout(data_dirs[config.scene_id], mp3dcat)

    with open(result_dir, mode='a') as file:
        file.write(f"map_miou: {map_miou}\n")
        file.write(f"map_accuracy: {map_accuracy}\n")
        file.write(f"lseg_miou: {lseg_miou}\n")
        file.write(f"lseg_accuracy: {lseg_accuracy}\n")

    print("map miou: ", map_miou)
    print("map accuracy: ", map_accuracy)
    print("lseg miou: ", lseg_miou)
    print("lseg accuracy: ", lseg_accuracy)


    




    

        

       
        


if __name__ == "__main__":
    main()
