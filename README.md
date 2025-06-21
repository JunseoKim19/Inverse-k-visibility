# Inverse k-visibility for RSSI-based Indoor Geometric Mapping

## Overview

This repository accompanies the paper:

**"Inverse k-visibility for RSSI-based Indoor Geometric Mapping"**  
by Junseo Kim, Matthew Lisondra, Yeganeh Bahoo, and Sajad Saeedi  


The paper introduces a novel algorithm for indoor geometric mapping using only WiFi signal strength (RSSI) data. The key innovation is the application of **inverse k-visibility**, a concept derived from computational geometry, which allows the inference of obstacle locations and free space without the use of traditional exteroceptive sensors like cameras or Lidar.


## How It Works

1. RSSI signals are collected as a robot traverses an indoor space.
2. Signal strength is converted into estimated k-values using a K-means clustering method.
3. The **inverse k-visibility** algorithm estimates free space and obstacles by analyzing changes in k-values along rays cast from known router positions.
4. The environment is mapped in real time using a probabilistic occupancy grid.

## Citation

If you use this work in your research, please cite:

```
@article{kim2025inverse,
  title={Inverse k-visibility for RSSI-based Indoor Geometric Mapping},
  author={Kim, Junseo and Lisondra, Matthew and Bahoo, Yeganeh and Saeedi, Sajad},
  journal={},
  year={2025},
  note={Submitted}
}
```

