

# Open-World Skill Discovery from Unsegmented Demonstrations

> :page_facing_up: [Read Paper](https://craftjarvis.github.io/SkillDiscovery/static/paper.pdf) \
  :link: [Project Website](https://craftjarvis.github.io/SkillDiscovery) \
  :hugs: [Datasets](https://huggingface.co/datasets/fatty-belly/MinecraftSkillDiscovery) \
  :space_invader: [MineRL Environment](https://github.com/minerllabs/minerl) (note version 1.0+ required)


# Environment Setup

Install pre-requirements for [MineRL](https://minerl.readthedocs.io/en/latest/tutorials/index.html).
Then install requirements with:

```
pip install git+https://github.com/minerllabs/minerl
pip install -r requirements.txt
```

# Generating Segmented Dataset

1. First download OpenAI's [VPT](https://github.com/openai/Video-Pre-Training) foundational action-prediction model and IDM model, and their dataset index file.
After that, download the dataset (the whole dataset is very HUGE so we only sample some data here, and you are free to change the sample number). You will see 3 directories under `data`: `videos`, `jsonl`(metadata), `zip`(compressed version of metadata).

```
sh download.sh
python download.py --sample_num 32 --worker_count 16
```

Notice that some video files are broken in OpenAI's website, so it is normal if the number of the downloaded files is less than the sample number.

2. Generate the segmented dataset.

For the loss-only and loss+info version:
```
python split.py --output_path ./result/loss_only.jsonl --num_cpus 32 --num_gpus 4
python split.py --output_path ./result/loss+info.jsonl --num_cpus 32 --num_gpus 4 --info
```

For the info-only version:
```
python tools/list_events.py --output_path ./result/info_only.jsonl
```

Now you can see the skill boundaries of the videos in the `result` directory.

3. Visualize the Length Distribution

You can visualize the length distribution of the 3 types of boundaries by running:
```
python tools/count_video_len.py
```

See the plots in `result` directory. The result will be slightly different from our paper because we use the whole dataset.

# Split Gameplay Videos

You can also try to split your own Minecraft gameplay videos! We also provide some videos in `example_videos`.

```
python segmentation.py --video_path ./example_videos/short.mp4
```

See the video clips in `result` directory (under the subdirectory with the same name as the video).

# Citation
```bibtex
@article{deng2025openworld,
  title={Open-World Skill Discovery from Unsegmented Demonstrations},
  author={Jingwen Deng and Zihao Wang and Shaofei Cai and Anji Liu and Yitao Liang},
  journal={arXiv preprint arXiv:2503.10684},
  year={2025}
}
```
