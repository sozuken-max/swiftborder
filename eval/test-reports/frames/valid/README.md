# Roboflow dataset v6 `valid` frames (camera 2701)

Upload the 15 `valid`-split images of Roboflow dataset v6 here: unannotated, 1920 x 1080 JPEG, as exported from `vehicle-detection-proejct` version 6. With the [`train/`](../train/README.md) folder and the 15 `test` frames one level up, this holds all 90 labelled images.

**Naming.** Rename each file from its Roboflow export name to the name in the `File` column. That is the same name as the GCS object and as `file` in [`annotations_with_predictions.json`](../../../runs/layer_a_count/annotations_with_predictions.json), which holds the box labels for these frames.
One image has no capture time in its name (its `File` is the Roboflow name without the `.rf.<hash>` suffix); keep that name.

**Status.** Not uploaded yet. When the files are in, check that every name below is present and update this line.

| # | File | Captured (SGT) | Cars | Buses | Trucks | Roboflow export name |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `2701_20250703T170545.jpg` | 2025-07-03 17:05:45 | 48 | 2 | 13 | `2701_20250703T170545_jpg.rf.9d69d7d65e34c376f71e7b948af8b8c4.jpg` |
| 2 | `2701_20250705T031544.jpg` | 2025-07-05 03:15:44 | 3 | 0 | 1 | `2701_20250705T031544_jpg.rf.0f4e1c1355068db3b7d21b797297c2a5.jpg` |
| 3 | `2701_20250705T090550.jpg` | 2025-07-05 09:05:50 | 119 | 5 | 7 | `2701_20250705T090550_jpg.rf.a2c1676a30f2e59be50c2ff3a38a78ac.jpg` |
| 4 | `2701_20250707T094559.jpg` | 2025-07-07 09:45:59 | 26 | 3 | 6 | `2701_20250707T094559_jpg.rf.57dc279d37fb78af6b8b554ce1840634.jpg` |
| 5 | `2701_20250708T050558.jpg` | 2025-07-08 05:05:58 | 1 | 0 | 0 | `2701_20250708T050558_jpg.rf.3eea89f6fc2adb268c76197cf1cc2b8f.jpg` |
| 6 | `2701_20250713T041602.jpg` | 2025-07-13 04:16:02 | 8 | 1 | 0 | `2701_20250713T041602_jpg.rf.293b93e3c5cb57255ad0857dfc51f4bc.jpg` |
| 7 | `2701_20250713T163533.jpg` | 2025-07-13 16:35:33 | 84 | 7 | 1 | `2701_20250713T163533_jpg.rf.f31213b5a286be8b739aa31ccf56d726.jpg` |
| 8 | `2701_20250715T055553.jpg` | 2025-07-15 05:55:53 | 14 | 11 | 8 | `2701_20250715T055553_jpg.rf.3c34c5bdb1ef6aef73020221a221e0ee.jpg` |
| 9 | `2701_20250716T114553.jpg` | 2025-07-16 11:45:53 | 20 | 1 | 22 | `2701_20250716T114553_jpg.rf.dcfc99af4783fc5cd49f3242f871130c.jpg` |
| 10 | `2701_20250722T180701.jpg` | 2025-07-22 18:07:01 | 16 | 20 | 14 | `2701_20250722T180701_jpg.rf.4340574e497a8827cff3db49a29b8a85.jpg` |
| 11 | `2701_20250726T115653.jpg` | 2025-07-26 11:56:53 | 69 | 6 | 13 | `2701_20250726T115653_jpg.rf.6d2238c564a74a5662f737b292d4a7d3.jpg` |
| 12 | `2701_20250729T041639.jpg` | 2025-07-29 04:16:39 | 4 | 0 | 0 | `2701_20250729T041639_jpg.rf.c28b6da04378a42c5b4e76552f53bb8f.jpg` |
| 13 | `2701_20250801T020708.jpg` | 2025-08-01 02:07:08 | 20 | 1 | 0 | `2701_20250801T020708_jpg.rf.eca1ae8a3e6f452c7a64d6ce82f179a8.jpg` |
| 14 | `2701_20251201T073621.jpg` | 2025-12-01 07:36:21 | 38 | 15 | 11 | `camera_id-2701_month-2025-12_2701_20251201T073621_jpg.rf.ca4fc4379783433e564f30c9c0b8e827.jpg` |
| 15 | `3dc0bb9d-49ce-4936-9e90-eb8b37e0a779.jpg` | unknown | 3 | 0 | 0 | `3dc0bb9d-49ce-4936-9e90-eb8b37e0a779_jpg.rf.9812aafbe13e485c806f8e110b367e4a.jpg` |

Totals: 473 cars, 72 buses, 96 trucks.
