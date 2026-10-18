# Data manifest (v4 runs)

SHA-256 and size of the raw files on the GPU host from which `scripts/03_preprocess_split.ipynb` built the processed data,
and of the split indices in `splits/`. Raw files are not redistributed (size and licensing); download them as described
in the README and compare with the hashes below.

- Edge-IIoTset: `ML-EdgeIIoT-dataset.csv` (157,800 records).
- CICIoT2023: merged CSV release `ciciot23.csv` (Kaggle `akashdogra/ciciot23csv`); every 31st record is kept (1,506,018).
- N-BaIoT: full version (Kaggle `mkashifn/nbaiot-dataset`): 89 traffic CSVs (one per capture) plus three auxiliary files
  (`data_summary.csv`, `device_info.csv`, `features.csv`) that are not traffic; class-stratified random sample of
  1,500,000 records (seed 42), split by capture file.
- `splits/<dataset>_split.npz`: row indices (`train`, `val`, `test`) into the processed table; the three sets are disjoint
  and together cover every record (sizes as in `results/split_report.json`).

| SHA-256 | Bytes | File |
|---|---:|---|
| `e75f67e1b309dce4d370c0f190640589b1e3199bf6dde469ffbbf830668208b0` | 5,217,440 | `splits/ciciot2023_split.npz` |
| `faffb5d7320af12a1ce781985ad43ccdedb7bc42d783e8153c69ce7565770342` | 265,129 | `splits/edge_iiotset_split.npz` |
| `5d8e67b0f4d982cdcacebe5bc8680ff7ff4533ed2c47a49ce67aed195a1ed4a9` | 2,288,923 | `splits/nbaiot_split.npz` |
| `c812fdc4b4fcd266b514d7f0cd320ed4b2ae46ed4deb1b6bd1c7d3117cad359a` | 13,754,096,319 | `data/raw/ciciot2023/ciciot23.csv` |
| `53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a` | 82,184,390 | `data/raw/edge_iiotset/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv` |
| `866083fd18f52b6ddde94e47b4d27925432bb83d5f547868f143d87c50354650` | 46,537,568 | `data/raw/nbaiot/1.benign.csv` |
| `99755635b255c1b05f1605259dbbf3f77200fc284930be888d4c3423a95815d9` | 105,833,909 | `data/raw/nbaiot/1.gafgyt.combo.csv` |
| `aafbfc32aef267dee69d9937a69e9561a636762a02a19bcbf437812325863842` | 49,322,665 | `data/raw/nbaiot/1.gafgyt.junk.csv` |
| `cca1095c1872c20a299fa522987f50bf9ec9828e2bd6983aef604ddc8a8eb2b5` | 40,177,901 | `data/raw/nbaiot/1.gafgyt.scan.csv` |
| `deb5e28f12424c4b8b5fd62706b7609bdaacf38a380b5eb3aa01e292f00af7c1` | 51,866,227 | `data/raw/nbaiot/1.gafgyt.tcp.csv` |
| `10f3cfff3c04016668a3a4904722bd9a9930bb17fbebaaf3b2dcd4df2b6e7fec` | 59,597,256 | `data/raw/nbaiot/1.gafgyt.udp.csv` |
| `ef411d0b9d9021c209e3c52cf83e3dd628ee3137c370c0a73c741dd3b0375483` | 135,498,914 | `data/raw/nbaiot/1.mirai.ack.csv` |
| `33184b64cf3edc279bd8001301409afb1f3b137f380cabf6d8519ee0343d085e` | 107,510,905 | `data/raw/nbaiot/1.mirai.scan.csv` |
| `58d4a359b9d0b1bd7f0ce49ba239e18fc0b27a361e8386413a61faec9d99c3e7` | 164,633,193 | `data/raw/nbaiot/1.mirai.syn.csv` |
| `bfc623dc77506276ff2af501ce02ecb40f6a6d812c3908fbab0a77776b29a490` | 315,304,260 | `data/raw/nbaiot/1.mirai.udp.csv` |
| `1f888f66c06dde773e1534cbcf35920717fbf35593583373b365c9b859afff2d` | 120,365,536 | `data/raw/nbaiot/1.mirai.udpplain.csv` |
| `87cc2a3d516491e92314d8a515dd2335cfe18c5dbef8a67328d78aed01f4dcfc` | 13,953,719 | `data/raw/nbaiot/2.benign.csv` |
| `2d2e95bd76c31d6efc17b47bde802ae711358df6d41e0ba9c19965f6f6dab14f` | 94,251,568 | `data/raw/nbaiot/2.gafgyt.combo.csv` |
| `f530a9c8c3582499121595c6bcfd04228dd807e1a20a3a5dcb7ab7c42fd4a258` | 51,673,314 | `data/raw/nbaiot/2.gafgyt.junk.csv` |
| `2f4f304637155627a083637e0a0b97dc7077f7ed810fb6f0a93874c629a086a7` | 37,108,696 | `data/raw/nbaiot/2.gafgyt.scan.csv` |
| `0283467266ee6d46d8a9208102246ab48783f47ef9878e5d5a7eb9bcb0a53c3d` | 53,485,288 | `data/raw/nbaiot/2.gafgyt.tcp.csv` |
| `b03f41406c92f655b517e6839a5dc4036bfd5206a5ee30cb0366cd1f624890a1` | 58,988,910 | `data/raw/nbaiot/2.gafgyt.udp.csv` |
| `1bed3daa5688fcc5ebf3d6b67c51cff43165d45bb97fab76eef0460d3fce5706` | 155,635,866 | `data/raw/nbaiot/2.mirai.ack.csv` |
| `504328ca668a7c762f756702f2498f6f6b9c70f68d690454f6198945454da8a0` | 43,078,827 | `data/raw/nbaiot/2.mirai.scan.csv` |
| `c7cec6b498eab4a99ff63b4f6ca80fc8a4be433f0a9d282ff9dcb03f1b284cac` | 161,570,382 | `data/raw/nbaiot/2.mirai.syn.csv` |
| `db06c4cbdc550727c9f2886767123b3551a41fb85181a264a6193cf49a695e52` | 207,783,110 | `data/raw/nbaiot/2.mirai.udp.csv` |
| `3654c1188fbc2b4c7b59fcc2d06a39042edce949396b2b098f968aadd1b9f341` | 138,671,843 | `data/raw/nbaiot/2.mirai.udpplain.csv` |
| `75cc608d3196950ad1cb14fc050babb96098128106b55ed84226e3ea97dba821` | 37,135,498 | `data/raw/nbaiot/3.benign.csv` |
| `8d943d53ba021d56721134b0582c9f3ddd2484c8bd7645230abd5a72df29290e` | 94,030,426 | `data/raw/nbaiot/3.gafgyt.combo.csv` |
| `5cc1a8a1615a30de172aaa0443d992a93bb32b30e38798a74968c152a8fffc75` | 50,828,262 | `data/raw/nbaiot/3.gafgyt.junk.csv` |
| `52e0f81933a4672251a7c62c6bdb67fe49c156245acd4b4ee15f9f2ec06838ae` | 37,971,404 | `data/raw/nbaiot/3.gafgyt.scan.csv` |
| `aebe5ac63d59408112bc292a5480617c02d815723201c94f7b9dad43dce5cf01` | 57,149,667 | `data/raw/nbaiot/3.gafgyt.tcp.csv` |
| `cd4777ed44ff01eff437dcb3458bec2b8ddb86db58879c32781bcffa384bcd64` | 58,507,137 | `data/raw/nbaiot/3.gafgyt.udp.csv` |
| `10a3508cbbbcd6b3e7647e570f0ab9c72f01f05c32660d2f25f537f9d4efe151` | 214,374,672 | `data/raw/nbaiot/4.benign.csv` |
| `923377bf736a67973fc4cd6b36455a44d07932b1b3a825d7ac5063069c8227b5` | 103,451,636 | `data/raw/nbaiot/4.gafgyt.combo.csv` |
| `3347890c08b2610e8dca0bb58d9406e9727520eb5308af35f1f84e392da7f746` | 48,260,845 | `data/raw/nbaiot/4.gafgyt.junk.csv` |
| `8eb3c88639feb9882126097cc743959e20ba59f25a3cdec079159364884bd792` | 37,535,324 | `data/raw/nbaiot/4.gafgyt.scan.csv` |
| `2eca9dc174682ba154b8c311b705b4acf2b0af82ff9c05f51b735416f3e7a315` | 52,113,658 | `data/raw/nbaiot/4.gafgyt.tcp.csv` |
| `76cceeac5043ca45939381a8d5c97dbcb8bac00d3293408a7cdf5b1cd908b7aa` | 59,545,509 | `data/raw/nbaiot/4.gafgyt.udp.csv` |
| `f78ec751df687bcba48ed79b30b4d945eeeb1d40de6fdf10bd05cbd0a29a4ee9` | 120,882,839 | `data/raw/nbaiot/4.mirai.ack.csv` |
| `4ed36577a5e0db1d31b8621dbf3227ae59ce8e674a8d852386e74abbef442513` | 103,452,295 | `data/raw/nbaiot/4.mirai.scan.csv` |
| `e715fb18f8d6279dab93760ab30587a595b2f99344fb55d754f250cf48715112` | 158,976,651 | `data/raw/nbaiot/4.mirai.syn.csv` |
| `cc1558f17896514d52ac9f23856a9beac0cec8369a2d93d58be99360599455f0` | 287,994,126 | `data/raw/nbaiot/4.mirai.udp.csv` |
| `8562521c9e85353dd32bb1fa616ce4d908a0f4231006c1955cf5a9df0ec195a6` | 118,674,570 | `data/raw/nbaiot/4.mirai.udpplain.csv` |
| `521cd6f18ecb53dee6c750b6139a035179109988f75aeb81fa26f549a8084ae8` | 64,576,152 | `data/raw/nbaiot/5.benign.csv` |
| `836bd15576c15b64fcfca36720ec3e3929328d6d362cc3cd829336b65aec92e7` | 109,076,363 | `data/raw/nbaiot/5.gafgyt.combo.csv` |
| `043ce520adc36255d4f36bde7b243fdf07f2171443c4b11d44ec9894f7ed9dcb` | 52,629,481 | `data/raw/nbaiot/5.gafgyt.junk.csv` |
| `7946a18b9336cc719e1658e568b470f46f243a2ec4fafcb2477f23afc76e57a1` | 39,580,175 | `data/raw/nbaiot/5.gafgyt.scan.csv` |
| `bb9233efb3497494fcb71f19199d49ce18b7013f2aa371b92b8d86d91388298a` | 58,822,798 | `data/raw/nbaiot/5.gafgyt.tcp.csv` |
| `bd9107b757128091288af71a69453136710907cc29f27313801274a0fc79eeaa` | 58,550,975 | `data/raw/nbaiot/5.gafgyt.udp.csv` |
| `3e38fe04f99c69cf7fd24b4ccb0cfa7c411b56cabfbb0d505fe13ce0506c9374` | 76,849,640 | `data/raw/nbaiot/5.mirai.ack.csv` |
| `2dcfcad287eff7ce72cc61a2c45e7f0d6888cbfd3eb8d177f59f15360f3e30c5` | 96,028,732 | `data/raw/nbaiot/5.mirai.scan.csv` |
| `d49b750bfa9c0178da5c517d180509bdc6a890485c11dbba8029e424ac1869cd` | 83,285,871 | `data/raw/nbaiot/5.mirai.syn.csv` |
| `568fc142fd90176ccde0953bc5c272499cf01bf0c268425c18a56591dedac978` | 200,357,797 | `data/raw/nbaiot/5.mirai.udp.csv` |
| `32ed5a5e34515d33a42f51132c1e5c03edf08b956f447ce68a7ddc92b5960597` | 79,817,584 | `data/raw/nbaiot/5.mirai.udpplain.csv` |
| `f897d6e347bd2cee9518c8aa6cdf873a793d505b341169383b52c7b1402affd3` | 103,620,766 | `data/raw/nbaiot/6.benign.csv` |
| `55f4a7c419113e0acd426ae7e5242725b5031e481af2eed9c815bb3be406a666` | 102,197,160 | `data/raw/nbaiot/6.gafgyt.combo.csv` |
| `c0b932340fb605b9e50f3d9acd1662995e81c2af127a141e911ee6509d495cad` | 49,536,332 | `data/raw/nbaiot/6.gafgyt.junk.csv` |
| `386b1753a817e3fd2241d005a3231703e5494d76e901b5f62f969c6a18ec24d2` | 38,452,514 | `data/raw/nbaiot/6.gafgyt.scan.csv` |
| `311c7d6d76284e3f7767656e9fe44291a6112dc64524c2c0b79475bfa73c1295` | 50,316,596 | `data/raw/nbaiot/6.gafgyt.tcp.csv` |
| `04b0cba05d2971df748ec742cbdd7dd1969172910cb86aece78fa1dd4f19cfe7` | 58,912,912 | `data/raw/nbaiot/6.gafgyt.udp.csv` |
| `3c514a706e57eae295d23d2cff00f895af9f2148f2f7214578f55b24078c56f9` | 73,553,957 | `data/raw/nbaiot/6.mirai.ack.csv` |
| `316919fe2550ffd8fbd1cf9312478f830a34437dae29dd2097cddf5466ab94fb` | 96,333,715 | `data/raw/nbaiot/6.mirai.scan.csv` |
| `5c0f18df7d4e50eb72c46bc0ad86d9d0b17cff3daa9c0458003e2f40f1a866c6` | 78,548,822 | `data/raw/nbaiot/6.mirai.syn.csv` |
| `c5fe69868536d87b5422e0c04d6f300f026adbb16c6a6797f3dae649a708883d` | 203,388,721 | `data/raw/nbaiot/6.mirai.udp.csv` |
| `6bfdf6d88ceeb1a397ab51afad0a372c0c2ac1e7937870946aa1e94e80328562` | 75,727,146 | `data/raw/nbaiot/6.mirai.udpplain.csv` |
| `b2f63cf6a42061de8d44f6d23be92a7e27c8ebb71edda80424b1897de304372f` | 54,969,458 | `data/raw/nbaiot/7.benign.csv` |
| `6571ae13c9eede42ca9f74f552c4599ce8003f37b18018dbd32cbce53a758748` | 104,015,943 | `data/raw/nbaiot/7.gafgyt.combo.csv` |
| `afd30eb66d4a82735a93b3a195412ffcf347710eb6463002984d8d75e4027d3d` | 48,261,599 | `data/raw/nbaiot/7.gafgyt.junk.csv` |
| `7660dca4cbc510a92ae7698b0321bbdd8e7b97eb0f70596291c9ba2f6c302921` | 37,441,442 | `data/raw/nbaiot/7.gafgyt.scan.csv` |
| `030c8ecec1525308aa533784d8035168b9a136e08e11fbcc104236ee5b2c0c7e` | 55,038,941 | `data/raw/nbaiot/7.gafgyt.tcp.csv` |
| `921beb8b9bf270614089eee524d21745787552010e7671330636955923f39aa4` | 62,265,519 | `data/raw/nbaiot/7.gafgyt.udp.csv` |
| `6ec7c1a83e1aa0ec510e63721e3e52d5f307b939c3807cd6accd284e8f6f6c43` | 47,616,941 | `data/raw/nbaiot/8.benign.csv` |
| `36114961849a84f39f5e0935f97ea67e8e69ccb1da3e01d13cc4cc6ca583215f` | 96,595,924 | `data/raw/nbaiot/8.gafgyt.combo.csv` |
| `4b55b24c90399fb51d0578bdc21de6183502b3903f5a215e158e2d1288e6d74b` | 48,585,877 | `data/raw/nbaiot/8.gafgyt.junk.csv` |
| `204fadeb472d67e47ecdcad0ad7faf317ef0037ed431cfcd19664704fe575068` | 37,511,054 | `data/raw/nbaiot/8.gafgyt.scan.csv` |
| `8d22555dd7db69573f62f71628b38670df3fb144219d51d97119a82b10235d96` | 49,995,960 | `data/raw/nbaiot/8.gafgyt.tcp.csv` |
| `83c3ea1de563b44d28c19309928039426b71c26a438c8e571b1281a5c32decfe` | 58,385,654 | `data/raw/nbaiot/8.gafgyt.udp.csv` |
| `30d66df604c4b46b417547d64e9f7e6fb9a1fa04cc4936aa4c2cf56ac9c78273` | 153,189,923 | `data/raw/nbaiot/8.mirai.ack.csv` |
| `17c6f5c00cadae95a55fca1a9e2829f3830b1cd8a14970c817a778f4e312a6a8` | 45,823,393 | `data/raw/nbaiot/8.mirai.scan.csv` |
| `d233b084dc3d9e508b4302a62094859985300a8dd2f99c41a53414e3ea757171` | 173,607,186 | `data/raw/nbaiot/8.mirai.syn.csv` |
| `d79e3ceb3a93bdf758852d2e497128cbe342f3474090b5abfe508dcebdbacbd5` | 208,331,925 | `data/raw/nbaiot/8.mirai.udp.csv` |
| `d5427a26d670848be77e1ae9ceeb5aa27e2920c958a9795001887bf3bc7cfcbb` | 124,485,105 | `data/raw/nbaiot/8.mirai.udpplain.csv` |
| `0e0f9c19bce92b3ebbf3173baff120d9ac99ffafb207618cbaea3eba53bcfac3` | 17,632,741 | `data/raw/nbaiot/9.benign.csv` |
| `3b0e8d6ecfe610a953d700da0e4b170b3a552513ace8c08d5d038169625acee4` | 105,589,038 | `data/raw/nbaiot/9.gafgyt.combo.csv` |
| `204e0933db90873f91954de666894a27c78c07d103fba2d1faa1d62d4fc1090f` | 46,573,306 | `data/raw/nbaiot/9.gafgyt.junk.csv` |
| `309d2d47b1360589bd3457b1d73033a2b4eeab275599363be01bf0eba909fdd6` | 38,455,148 | `data/raw/nbaiot/9.gafgyt.scan.csv` |
| `22608905531b663482dc45c66d95199a879e006cd171284bf16f7a72bdaf634e` | 55,203,340 | `data/raw/nbaiot/9.gafgyt.tcp.csv` |
| `0905dbd1ff34eaad2b4e977fe58ffe0e3aab81e79b2499cc295f2cd5e327d607` | 57,969,199 | `data/raw/nbaiot/9.gafgyt.udp.csv` |
| `67434b1b64e3e27fa3415732f749e4b1c4c099485c156f5aca3c7c0790cf4ff9` | 147,187,372 | `data/raw/nbaiot/9.mirai.ack.csv` |
| `bc15d2a4833b9f12f51837c6581fb55c6cf8ed13bc8699c77dcd29c3988b359d` | 43,554,193 | `data/raw/nbaiot/9.mirai.scan.csv` |
| `9545795b938ef0ad6744154dd47fe7245e16f45aaebb4fa5ed5196859f035498` | 169,089,092 | `data/raw/nbaiot/9.mirai.syn.csv` |
| `bb6f6958b74be6a4055f8795950deefcfd41e4447c04ca48555d2adeae5418a3` | 215,629,797 | `data/raw/nbaiot/9.mirai.udp.csv` |
| `606bb98a6ffd746e2a33dffc373e8910762f81a90048d2d497a7e0734ed594fb` | 133,920,179 | `data/raw/nbaiot/9.mirai.udpplain.csv` |
| `b6403968bdc1326571f1bb2a188cd935c8082a19d83bdb8322e310117f243324` | 2,567 | `data/raw/nbaiot/data_summary.csv` |
| `7da79236d5935ef2eeff7d6d370fa4733aef405be39f99940234e56bb006f70c` | 302 | `data/raw/nbaiot/device_info.csv` |
| `385cd9783c75d6dd0e7211a3e59705d953075ebaf19325dfcbe54d67f80ae930` | 10,979 | `data/raw/nbaiot/features.csv` |
