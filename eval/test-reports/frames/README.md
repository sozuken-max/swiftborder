# Labelled frames (camera 2701)

All 90 labelled frames of Roboflow dataset v6, unannotated, 1920 x 1080 JPEG, as three zips. The labels are in [`annotations_with_predictions.json`](../../runs/layer_a_count/annotations_with_predictions.json) (all 90 images, with the YOLO-S predictions). Unzip before use; the zips hold the files at their top level.

| File | Contents |
| --- | --- |
| [`test_frames15.zip`](test_frames15.zip) | The 15 `test` frames (described below) |
| [`train_frames60.zip`](train_frames60.zip) | The 60 `train` frames ([list](#train-frames-60)) |
| [`valid_frames15.zip`](valid_frames15.zip) | The 15 `valid` frames ([list](#valid-frames-15)) |

The rest of this page describes the 15 `test` frames. These are the frames behind every count and box score in the [test report](../README.md). They were not used to train any of the fine-tuned detectors or to select their training checkpoint epoch (note that the comparison notebook's per-model confidence threshold in the artifact table was swept on these 15 frames, whereas the served `model=local` operating point is fixed at `conf=0.20, overlap=0.60`).

**Naming.** `2701_<YYYYMMDD>T<HHMMSS>.jpg`: camera id, then the capture time in SGT. This is the same name as the object in `gs://sg-lta-traffic-cameras/camera_id=2701/month=<YYYY-MM>/`. Roboflow renamed the files on export (`..._jpg.rf.<hash>.jpg`), and that name is what the [notebooks](../notebooks/), `test_counts.csv` and the Drive `test_predictions/` folders use; the table maps one to the other.

**Where the bytes came from.** Downloaded on 17 Oct 2026 from the data.gov.sg URL in the last column (the URL the live `swiftbackend` resolved for that timestamp). The 03:15 frame was earlier checked byte-identical to its GCS object; the other 14 were not compared. The box labels are in the Roboflow project (`vehicle-detection-proejct`, version 6); an export of them is in [`annotations_with_predictions.json`](../../runs/layer_a_count/annotations_with_predictions.json).

**Day / night.** Day is 07:00-18:59 SGT (7 frames), night the other 8.

| # | File | Captured (SGT) | Day/night | GT cars | GT vehicles | Roboflow export name | sha256 (first 12) | data.gov.sg source |
| ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | [`2701_20250701T031541.jpg`](2701_20250701T031541.jpg) | 01 Jul 2025 03:15:41 | night | 3 | 5 | `2701_20250701T031541_jpg.rf.305d8ff6133a3042f867f23ec741ae18.jpg` | `fe23db91ae23` | [link](https://images.data.gov.sg/api/traffic-images/2025/06/8e8a4b97-d682-40ee-b6a9-17b4bc8854dd.jpg) |
| 2 | [`2701_20250701T104609.jpg`](2701_20250701T104609.jpg) | 01 Jul 2025 10:46:09 | day | 75 | 80 | `camera_id-2701_month-2025-07_2701_20250701T104609_jpg.rf.ac3643752e84aa54a25355ffb165f008.jpg` | `ba3146ab3b9a` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/235998c9-ffab-422f-850b-c284674f6a66.jpg) |
| 3 | [`2701_20250701T171615.jpg`](2701_20250701T171615.jpg) | 01 Jul 2025 17:16:15 | day | 33 | 36 | `2701_20250701T171615_jpg.rf.2c614559e4c9ff55ed03ec2de9d135e9.jpg` | `8c3f83be6fcf` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/23a325c9-7a70-424d-b26c-eaf295cf864d.jpg) |
| 4 | [`2701_20250707T004549.jpg`](2701_20250707T004549.jpg) | 07 Jul 2025 00:45:49 | night | 9 | 14 | `2701_20250707T004549_jpg.rf.c77044050d670fc2bfffa93675c6da80.jpg` | `2197d035f021` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/612339b4-8855-45c7-9973-8f6b09bef2a3.jpg) |
| 5 | [`2701_20250711T045532.jpg`](2701_20250711T045532.jpg) | 11 Jul 2025 04:55:32 | night | 8 | 12 | `2701_20250711T045532_jpg.rf.678d451f4415b965cd680ea85ad15882.jpg` | `0896d506da5c` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/2e6ac41f-217e-441a-ba12-f961c3f2e98c.jpg) |
| 6 | [`2701_20250714T001542.jpg`](2701_20250714T001542.jpg) | 14 Jul 2025 00:15:42 | night | 66 | 66 | `2701_20250714T001542_jpg.rf.3090aa01df1162d781004f6e01eddaee.jpg` | `d5b7ec3e0a09` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/7eaa82d5-c4a7-4e88-9772-249c754fb8f3.jpg) |
| 7 | [`2701_20250714T105553.jpg`](2701_20250714T105553.jpg) | 14 Jul 2025 10:55:53 | day | 56 | 72 | `2701_20250714T105553_jpg.rf.f73246c6a77913a22e247d3b2ae89c6c.jpg` | `1f8d93e6cc2d` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/a4373f3d-2e5e-442f-a934-60db65e0b43a.jpg) |
| 8 | [`2701_20250715T014608.jpg`](2701_20250715T014608.jpg) | 15 Jul 2025 01:46:08 | night | 0 | 1 | `2701_20250715T014608_jpg.rf.a2c117d8afe7ba47b46288008083e93b.jpg` | `893c62c018ff` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/f96b6c7d-d2a9-4ab6-bdb1-e0a143d93a65.jpg) |
| 9 | [`2701_20250715T144541.jpg`](2701_20250715T144541.jpg) | 15 Jul 2025 14:45:41 | day | 10 | 15 | `2701_20250715T144541_jpg.rf.35b63305e56bfae4ea16bc6ad81ca58d.jpg` | `065bb1bee372` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/764579a4-75e5-4cf2-a7c5-53c2c9d7b73c.jpg) |
| 10 | [`2701_20250718T182605.jpg`](2701_20250718T182605.jpg) | 18 Jul 2025 18:26:05 | day | 39 | 45 | `2701_20250718T182605_jpg.rf.809298121812af11927ea2b6fb7b096c.jpg` | `f8d0f6e462cb` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/aa07e6b4-0fa5-4c03-99e9-4a8c09f2c694.jpg) |
| 11 | [`2701_20250719T093542.jpg`](2701_20250719T093542.jpg) | 19 Jul 2025 09:35:42 | day | 87 | 91 | `2701_20250719T093542_jpg.rf.40c97a80db5f5b7eb25b18ffc2665168.jpg` | `f70960b8ec2d` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/44552e5a-1a1c-4dd3-8df4-495e1290c614.jpg) |
| 12 | [`2701_20250720T021638.jpg`](2701_20250720T021638.jpg) | 20 Jul 2025 02:16:38 | night | 1 | 3 | `2701_20250720T021638_jpg.rf.0209baf89dfb6e4eb51b3a9414f482db.jpg` | `4d715c6749c2` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/fb3c747e-754f-4a8d-bc3f-ef6439ac5ae6.jpg) |
| 13 | [`2701_20250722T074652.jpg`](2701_20250722T074652.jpg) | 22 Jul 2025 07:46:52 | day | 24 | 51 | `2701_20250722T074652_jpg.rf.726f91de5a9c04d4507b0be46bb14f17.jpg` | `4472b19ae4b0` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/f4fa0d10-f6d5-489e-adcc-2476a25fa568.jpg) |
| 14 | [`2701_20250722T195703.jpg`](2701_20250722T195703.jpg) | 22 Jul 2025 19:57:03 | night | 15 | 26 | `2701_20250722T195703_jpg.rf.dc042f89e0c9ca98ebdc67197f07a798.jpg` | `79fd9df0187b` | [link](https://images.data.gov.sg/api/traffic-images/2025/07/ee24c56b-fd4e-4b5d-b864-9d8e41cd1feb.jpg) |
| 15 | [`2701_20250901T190618.jpg`](2701_20250901T190618.jpg) | 01 Sep 2025 19:06:18 | night | 25 | 29 | `camera_id-2701_month-2025-09_2701_20250901T190618_jpg.rf.646c1fea1ba2d48f432680f3cf43ff09.jpg` | `85750d77d8ac` | [link](https://images.data.gov.sg/api/traffic-images/2025/09/efc26406-85d1-4858-9aee-435e7beb5f45.jpg) |

Totals: 451 cars and 546 vehicles, matching the split's label counts in the notebook (451 cars, 25 buses, 70 trucks).

The speed test's recorded GPU run used 20 live frames that were not kept, so they are not here. Rerun it on these frames with `--frames` (see [Speed test](../README.md#speed-test)).

**Zip copy.** `test_frames15.zip` was written on a Mac and its JPEGs are re-encoded (same 1920 x 1080 pixels, different bytes), so the sha256 column above is the hash of the original files, not of the zip contents. The 03:15 frame's byte-match to its GCS object was checked on the original, not on the zip copy. The zips contain no `__MACOSX` data that matters; ignore that folder if your unzip tool extracts it.

## Train frames (60)

File names are `2701_<YYYYMMDD>T<HHMMSS>.jpg`; the last column is the Roboflow export name used in the notebooks. Totals: 1832 cars, 170 buses, 284 trucks.

| # | File | Captured (SGT) | Cars | Buses | Trucks | Roboflow export name |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `2701_20250701T021540.jpg` | 2025-07-01 02:15:40 | 6 | 0 | 6 | `2701_20250701T021540_jpg.rf.bf1c7b1f9b645a45e303f870775f27b0.jpg` |
| 2 | `2701_20250701T145553.jpg` | 2025-07-01 14:55:53 | 9 | 0 | 5 | `2701_20250701T145553_jpg.rf.ac8cda06abdb59a053bc6b283fb4ecf2.jpg` |
| 3 | `2701_20250702T132537.jpg` | 2025-07-02 13:25:37 | 6 | 1 | 29 | `2701_20250702T132537_jpg.rf.4cbb51458c384710f18b432b62a87f91.jpg` |
| 4 | `2701_20250702T222526.jpg` | 2025-07-02 22:25:26 | 16 | 1 | 0 | `2701_20250702T222526_jpg.rf.3881ba41df7f9efaba4a2faf9b1ff3c4.jpg` |
| 5 | `2701_20250703T032552.jpg` | 2025-07-03 03:25:52 | 5 | 0 | 0 | `2701_20250703T032552_jpg.rf.2ecd8664ce0ebdb61956248e0fafbbf4.jpg` |
| 6 | `2701_20250703T180545.jpg` | 2025-07-03 18:05:45 | 65 | 13 | 14 | `2701_20250703T180545_jpg.rf.cea764e3cf42f847bdc3ba92bcd5128e.jpg` |
| 7 | `2701_20250703T184606.jpg` | 2025-07-03 18:46:06 | 12 | 4 | 3 | `2701_20250703T184606_jpg.rf.a702232c3133ff26d3f50eee1767d86c.jpg` |
| 8 | `2701_20250704T164552.jpg` | 2025-07-04 16:45:52 | 12 | 0 | 3 | `2701_20250704T164552_jpg.rf.f7de308555b42f52ff27dd7b8f3fe013.jpg` |
| 9 | `2701_20250705T051546.jpg` | 2025-07-05 05:15:46 | 18 | 1 | 1 | `2701_20250705T051546_jpg.rf.973f4b471ca681fb7d6b0be71a911d8f.jpg` |
| 10 | `2701_20250706T060552.jpg` | 2025-07-06 06:05:52 | 10 | 2 | 0 | `2701_20250706T060552_jpg.rf.156dd62d4c04cce7e4d860f747d0023e.jpg` |
| 11 | `2701_20250706T112537.jpg` | 2025-07-06 11:25:37 | 104 | 9 | 1 | `2701_20250706T112537_jpg.rf.abc7910976224712d404bbaf7698ec79.jpg` |
| 12 | `2701_20250707T002548.jpg` | 2025-07-07 00:25:48 | 9 | 0 | 1 | `2701_20250707T002548_jpg.rf.b05ac19f266305e649da65274ba0505a.jpg` |
| 13 | `2701_20250707T013530.jpg` | 2025-07-07 01:35:30 | 7 | 0 | 0 | `2701_20250707T013530_jpg.rf.524bda4cc7719045930d1202a07d3470.jpg` |
| 14 | `2701_20250708T105544.jpg` | 2025-07-08 10:55:44 | 59 | 2 | 8 | `2701_20250708T105544_jpg.rf.63e840837b4dd12f6ac78af6739ae75d.jpg` |
| 15 | `2701_20250708T131606.jpg` | 2025-07-08 13:16:06 | 4 | 1 | 19 | `2701_20250708T131606_jpg.rf.862fdb47e0ece1887371388b450d2310.jpg` |
| 16 | `2701_20250709T002518.jpg` | 2025-07-09 00:25:18 | 10 | 0 | 0 | `2701_20250709T002518_jpg.rf.00fdcb60bb73cc9ae4c9cf69442a7126.jpg` |
| 17 | `2701_20250709T124551.jpg` | 2025-07-09 12:45:51 | 84 | 3 | 12 | `2701_20250709T124551_jpg.rf.5cb1301320fa9af45046d560b6f27c78.jpg` |
| 18 | `2701_20250709T132551.jpg` | 2025-07-09 13:25:51 | 58 | 1 | 2 | `2701_20250709T132551_jpg.rf.ada9eef51a6b8eb77d8b57c5c3e3ee58.jpg` |
| 19 | `2701_20250710T000602.jpg` | 2025-07-10 00:06:02 | 12 | 1 | 0 | `2701_20250710T000602_jpg.rf.34410ecb022130fff493562ed7221996.jpg` |
| 20 | `2701_20250710T051608.jpg` | 2025-07-10 05:16:08 | 5 | 2 | 1 | `2701_20250710T051608_jpg.rf.edb2697cc65b3d895bdec4c19c2781e8.jpg` |
| 21 | `2701_20250710T140556.jpg` | 2025-07-10 14:05:56 | 8 | 1 | 8 | `2701_20250710T140556_jpg.rf.482ae12426f3d88cbcd1b842f4d8d7c0.jpg` |
| 22 | `2701_20250711T203527.jpg` | 2025-07-11 20:35:27 | 43 | 1 | 0 | `2701_20250711T203527_jpg.rf.7fc21d21c586271b6224bc839d04c2fd.jpg` |
| 23 | `2701_20250712T111543.jpg` | 2025-07-12 11:15:43 | 38 | 1 | 22 | `2701_20250712T111543_jpg.rf.a65e77797e4f111045f293101ee9968e.jpg` |
| 24 | `2701_20250712T112543.jpg` | 2025-07-12 11:25:43 | 99 | 2 | 6 | `2701_20250712T112543_jpg.rf.da86b88a0b05427342e2b157118dbc73.jpg` |
| 25 | `2701_20250712T172548.jpg` | 2025-07-12 17:25:48 | 39 | 3 | 0 | `2701_20250712T172548_jpg.rf.443b9185b3561c9a34e7934d9a5d9640.jpg` |
| 26 | `2701_20250712T204532.jpg` | 2025-07-12 20:45:32 | 10 | 1 | 1 | `2701_20250712T204532_jpg.rf.ae73f654dc03ca5e0a0e9b5f4f652c7b.jpg` |
| 27 | `2701_20250713T153552.jpg` | 2025-07-13 15:35:52 | 46 | 4 | 0 | `2701_20250713T153552_jpg.rf.7356eff0c8c7adcf00299e0f580134c3.jpg` |
| 28 | `2701_20250713T200537.jpg` | 2025-07-13 20:05:37 | 13 | 0 | 0 | `2701_20250713T200537_jpg.rf.858a6fdd56e700f988cc646e422be254.jpg` |
| 29 | `2701_20250714T095532.jpg` | 2025-07-14 09:55:32 | 13 | 0 | 18 | `2701_20250714T095532_jpg.rf.c5e928cfaa1fb4cf9c10d87ec8abd2bb.jpg` |
| 30 | `2701_20250714T210603.jpg` | 2025-07-14 21:06:03 | 14 | 0 | 0 | `2701_20250714T210603_jpg.rf.0d4bb18c501e2ce006a7f5c353d4053f.jpg` |
| 31 | `2701_20250715T011547.jpg` | 2025-07-15 01:15:47 | 5 | 0 | 1 | `2701_20250715T011547_jpg.rf.a2fb18d7653df58e235b94080c80baa4.jpg` |
| 32 | `2701_20250715T042551.jpg` | 2025-07-15 04:25:51 | 2 | 0 | 2 | `2701_20250715T042551_jpg.rf.f3d9b83ba773f56098b2fe11bbcca62d.jpg` |
| 33 | `2701_20250715T173533.jpg` | 2025-07-15 17:35:33 | 80 | 13 | 8 | `2701_20250715T173533_jpg.rf.6c7d116773dd2b6ab0b619b62c12eea6.jpg` |
| 34 | `2701_20250716T212541.jpg` | 2025-07-16 21:25:41 | 17 | 2 | 1 | `2701_20250716T212541_jpg.rf.968f07fa530062cadab58204c1e3f4a7.jpg` |
| 35 | `2701_20250717T005525.jpg` | 2025-07-17 00:55:25 | 4 | 0 | 1 | `2701_20250717T005525_jpg.rf.8f34f3a162b353069eff622b0cc8a84d.jpg` |
| 36 | `2701_20250717T062552.jpg` | 2025-07-17 06:25:52 | 30 | 22 | 1 | `2701_20250717T062552_jpg.rf.da7f53ebc1833bbf59eb254067a0275c.jpg` |
| 37 | `2701_20250718T163603.jpg` | 2025-07-18 16:36:03 | 70 | 2 | 3 | `2701_20250718T163603_jpg.rf.9d498ee47d143be3d77e3fe591a3ae67.jpg` |
| 38 | `2701_20250718T220529.jpg` | 2025-07-18 22:05:29 | 31 | 3 | 5 | `2701_20250718T220529_jpg.rf.fcc756cc4812adfe96d58c40f9527a95.jpg` |
| 39 | `2701_20250719T004552.jpg` | 2025-07-19 00:45:52 | 8 | 4 | 1 | `2701_20250719T004552_jpg.rf.9625c1ff995f2fa28e4e517d17b88f45.jpg` |
| 40 | `2701_20250720T000555.jpg` | 2025-07-20 00:05:55 | 96 | 5 | 0 | `2701_20250720T000555_jpg.rf.5490127437457879623b606ec145ae1f.jpg` |
| 41 | `2701_20250720T013637.jpg` | 2025-07-20 01:36:37 | 40 | 0 | 1 | `2701_20250720T013637_jpg.rf.47272d084c4bde445db9be936775a2d0.jpg` |
| 42 | `2701_20250720T014637.jpg` | 2025-07-20 01:46:37 | 3 | 0 | 2 | `2701_20250720T014637_jpg.rf.4a03b04e7d0cb985f6ac38d23bb958f6.jpg` |
| 43 | `2701_20250720T181653.jpg` | 2025-07-20 18:16:53 | 16 | 1 | 1 | `2701_20250720T181653_jpg.rf.c57e28305db46929cd6a3df87d67de66.jpg` |
| 44 | `2701_20250720T182653.jpg` | 2025-07-20 18:26:53 | 25 | 1 | 0 | `2701_20250720T182653_jpg.rf.4159eabd23734bdc20f9ab11ca5ed8d2.jpg` |
| 45 | `2701_20250721T185658.jpg` | 2025-07-21 18:56:58 | 104 | 5 | 2 | `2701_20250721T185658_jpg.rf.ba61b300fe826aab9344038cb443f0b5.jpg` |
| 46 | `2701_20250723T001708.jpg` | 2025-07-23 00:17:08 | 10 | 1 | 0 | `2701_20250723T001708_jpg.rf.1b281243bed226cb9e9b6b2d86090095.jpg` |
| 47 | `2701_20250723T085638.jpg` | 2025-07-23 08:56:38 | 51 | 14 | 5 | `2701_20250723T085638_jpg.rf.ff492cffd9f88b598794651fb3124784.jpg` |
| 48 | `2701_20250723T094638.jpg` | 2025-07-23 09:46:38 | 23 | 0 | 19 | `2701_20250723T094638_jpg.rf.022a8a95013584752ccd42ea059b4a7a.jpg` |
| 49 | `2701_20250724T025636.jpg` | 2025-07-24 02:56:36 | 2 | 0 | 0 | `2701_20250724T025636_jpg.rf.19e5cadf91c96dcff049d365d2046e65.jpg` |
| 50 | `2701_20250725T001656.jpg` | 2025-07-25 00:16:56 | 10 | 0 | 0 | `2701_20250725T001656_jpg.rf.49527e18d22062ae52c0cce64f4e050e.jpg` |
| 51 | `2701_20250725T133710.jpg` | 2025-07-25 13:37:10 | 52 | 6 | 19 | `2701_20250725T133710_jpg.rf.b3a488e3fb109144fb83bf76ee3b215b.jpg` |
| 52 | `2701_20250726T065648.jpg` | 2025-07-26 06:56:48 | 17 | 1 | 4 | `2701_20250726T065648_jpg.rf.02d65255012ed0cd17f6d8998f6d2e6a.jpg` |
| 53 | `2701_20250727T051653.jpg` | 2025-07-27 05:16:53 | 9 | 0 | 0 | `2701_20250727T051653_jpg.rf.cb35f5b0009feda67d7ff118fabab970.jpg` |
| 54 | `2701_20250729T070702.jpg` | 2025-07-29 07:07:02 | 52 | 4 | 19 | `2701_20250729T070702_jpg.rf.081e93958db044e3cf712fb29a50fe4b.jpg` |
| 55 | `2701_20250729T092704.jpg` | 2025-07-29 09:27:04 | 26 | 2 | 13 | `2701_20250729T092704_jpg.rf.df05ac6b1ba6e6003dbc0ae7adcc49dd.jpg` |
| 56 | `2701_20250729T140708.jpg` | 2025-07-29 14:07:08 | 9 | 1 | 3 | `2701_20250729T140708_jpg.rf.118adbe8c6130f16d4acef23d5cad0f7.jpg` |
| 57 | `2701_20250729T182711.jpg` | 2025-07-29 18:27:11 | 94 | 16 | 6 | `2701_20250729T182711_jpg.rf.c13832ac2f9278f99c450a88f5e6e521.jpg` |
| 58 | `2701_20250730T180655.jpg` | 2025-07-30 18:06:55 | 91 | 9 | 6 | `2701_20250730T180655_jpg.rf.8744010fa04eb3669c5e45b0613a6450.jpg` |
| 59 | `2701_20250901T162556.jpg` | 2025-09-01 16:25:56 | 12 | 3 | 0 | `camera_id-2701_month-2025-09_2701_20250901T162556_jpg.rf.6db3df3fca84005b333341451ea09413.jpg` |
| 60 | `2701_20250901T194559.jpg` | 2025-09-01 19:45:59 | 9 | 1 | 1 | `camera_id-2701_month-2025-09_2701_20250901T194559_jpg.rf.9f5ece57bd678905646d85c275485197.jpg` |

## Valid frames (15)

One image (#15) has no capture time in its name; in the zip it keeps its full Roboflow export name (`3dc0bb9d-...-9e90-eb8b37e0a779_jpg.rf.9812aafbe13e485c806f8e110b367e4a.jpg`), not the short name below. Totals: 473 cars, 72 buses, 96 trucks.

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
