/**
  ******************************************************************************
  * @file    predictive_maintenance_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-03-19T09:49:17+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "predictive_maintenance_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_predictive_maintenance_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_predictive_maintenance_weights_array_u64[1571] = {
  0x3e2519eabdf927b4U, 0xbd404ad33ea5a644U, 0x3eebb309bdeb4d42U, 0xbd0519a1beffe382U,
  0x3c4328bfbbd77561U, 0xbe94b977befac1b7U, 0x3ea2139cbef52a80U, 0x3e79fec13ed4f80fU,
  0x3e4dc5a0beae79f2U, 0x3d1d4e113e3fca3aU, 0xbe133fa73e6f955dU, 0xbe38df5fbf0ce99bU,
  0xbe29ec843e7a4279U, 0x3e9f12a7bea06f57U, 0xbdf91521be2ffedaU, 0xbe6cae503ddc4d9bU,
  0xbef9d30a3dbf6a5dU, 0x3bfcc1793e7f9fe2U, 0x3d44abcebdb58a71U, 0xbc14c69b3d6bfb44U,
  0x3ebd07a53e2fd1caU, 0x3d5ca7aa3daed98dU, 0xbefc0fadbeba5fa3U, 0xbdac2c453e14af2fU,
  0x3e6e4d69be0f2cd9U, 0xbeb07e59be9e8e91U, 0xbe5ed0a1bb82eac9U, 0xbe4446b23d70bcd9U,
  0xbec820183db29a4aU, 0xbcfe6c803e099909U, 0x3e066674be210d72U, 0x3e9e3f603eebc689U,
  0x3e742bd1be017d27U, 0xbe7d32f53efd8b8cU, 0x3cfdf02fbe3aa40eU, 0xbdb5dc45bee32b03U,
  0xbeb5f7963f0167a1U, 0xbc4071ffbe85900bU, 0xbe583edebe74a77dU, 0x3cafe008bdc54d1fU,
  0x3e8b60d43dbfb3b0U, 0x3de104023e465f9dU, 0x3df48072be2beb2aU, 0x3ddfaaec3ca19206U,
  0xbd285e8b3ec4e71dU, 0x3d0e00bebe3e27b4U, 0x3ea77fd73e86488aU, 0x3e5ac8713e5e51f7U,
  0x3e20e141be66a6b8U, 0x3e10327e3f02cbd9U, 0x3ec323c5be39b4b1U, 0xbdff9256bad8a3e8U,
  0xbe6d1dce3e38e58fU, 0xbdef09e73dc37bb2U, 0xbee48a1d3eff7dcfU, 0x3df9734e3e9874adU,
  0x3d1ed6e1befdd526U, 0x3db6b431b840bd65U, 0xbf0e41d9bea63b2eU, 0xbd716d333d6d6951U,
  0xbe94f0ca3e958ba6U, 0xbe2a55e03dbc933bU, 0xbe9f0127bef4487aU, 0xbee11551be916eecU,
  0x3baafee33e4604e9U, 0xbdf9704bbf1b8e4eU, 0x3d3c554ebe1a2452U, 0x3e32c1f23e1b60e1U,
  0x3e80957b3e776c3bU, 0xbe8f7ab23e558c8aU, 0x3e48fbeb3e67f918U, 0xbe99e0eabe97d610U,
  0x3dfbcc29be096b77U, 0xbe519c313ce07192U, 0x3e82c7c53eb627acU, 0x3e13bc9cbdd97b5bU,
  0xbefd4d4fbe337856U, 0x3c16d6193e068e77U, 0x3e5d5dbf3d3b1945U, 0xbeacd6ba3de0c031U,
  0x3db1bfa1bf067c04U, 0xbe9c5b603f134b76U, 0xbb9d130ebe2faa1bU, 0xbe651499be540625U,
  0x3d64c439be2f73b4U, 0x3e5006fc3e73cdceU, 0x3eb50a593e32c93cU, 0x3cf6260bbc5a6cc4U,
  0xbefff724be380b50U, 0x3c2e8b813dbae291U, 0xbe92dd303eb5af82U, 0xbcf15e693e735fabU,
  0xbea47a25bf064c6cU, 0x3e4488a2be2e7d2bU, 0x3caced533ef88b93U, 0x3f008c54be5c4757U,
  0xbdedcf4ebb842eafU, 0xbee3d4853d959094U, 0xbdae39553e1074a6U, 0x3dbfb034bdbd473eU,
  0xbf056387be4b8245U, 0xbd0b0d353e2c959eU, 0x3d6528d3be21e1deU, 0x3e2fb07a3e6771e6U,
  0x3e4fd6623e5e0b6fU, 0x3ea07accbe8468bbU, 0x3e6b704b3eb25373U, 0x3e884a7d3d914787U,
  0xbe2e2f023e9cf4aaU, 0xbdd04ba33e5bd13cU, 0xbe15e575bf0f12cbU, 0x3ea1416bbeabfc53U,
  0x3c447afb3edba994U, 0x3ef482ebbe955a54U, 0xbd92e9f4bd9d8f5eU, 0xbe9b56fd3e50f4e7U,
  0xbe8053553e6b9bcaU, 0x3cb8915bbdcbd9bfU, 0xbeb840713e4e2c02U, 0xbdff88543e6125b5U,
  0xbe20c31e3ea9bc9aU, 0xbd5d9198bf012b2dU, 0xbe771eccbea11ec8U, 0x3e92d8ad3e1cfd8dU,
  0xbe421ead3e4ab284U, 0xbd6393fdbec94302U, 0xbf37e7463dfc9a31U, 0xbe0c4ac53e9e2ff0U,
  0x3e3db5a6bf03d5feU, 0x3e061ffbbe477924U, 0x3dd33bf73dba7e3aU, 0x3a7a18fb3ecae1f3U,
  0xbe6f5a903e277ea7U, 0xbee929ee3e14b25dU, 0x3b78fc583e49c457U, 0xbe9cb3ee3eca4ddeU,
  0xbd1ec5763e3651fcU, 0x3e2aaababefb6daaU, 0xbea691063f0db1ffU, 0x3cf4c88fbe82006dU,
  0xbe87ce26be662e7bU, 0x3e26c7dbbe65fbe8U, 0x3dcccf1c3d6037feU, 0x3c6cc4b73ed5ce61U,
  0x3e6b35cebee2436dU, 0x3e3967103dcc47c6U, 0x3e8561fb3d3351e7U, 0x3d8556fbbda52fb9U,
  0x3e5e53753cdac507U, 0xbcd603cc3e9c3d72U, 0xbe3366403eb5019eU, 0xbef2693c3e401e84U,
  0x3e7e465dbc1f5e75U, 0xbe7fe6253ec3d0f4U, 0xbd9acad1beeafab8U, 0xbdc7644dbe479e51U,
  0xbec416443f138f8eU, 0x3d1d628dbeb49c03U, 0xbe70d734bea83239U, 0x3e09b4d6bf0202c1U,
  0x3d52e9fc3e1b5a3bU, 0x3e178659bed8cd8cU, 0x3e2e8342bde7a879U, 0xbd05ace73e6d2ad2U,
  0x3f0a0de7bea35a24U, 0x3eb5b969beb0df54U, 0xbec52282be37c958U, 0x3dc0198fbe75f525U,
  0x3de86e5fbe1b5d81U, 0x3db165e33d2d51a1U, 0x3cd563123eb01fcdU, 0x3c18ef923e5071d8U,
  0xbe0435573e6efb48U, 0xbe33a301bf0bd2bdU, 0x3e6ffd87bde3de90U, 0xbf13a52fbe857369U,
  0xbdb2ee883d9871f8U, 0x3daa3d40bdadc734U, 0x3eb1ccac3f05d433U, 0x3e551d97be0985f5U,
  0x3c8fd8113e3e4fe9U, 0xbe9a1fefbedc5ff3U, 0xbeab92cc3cf037faU, 0x3d157e71be0a799eU,
  0x3e6a5f9e3e1f0d61U, 0x3df1af9a3e4bd2dbU, 0x3e21da8cbe08093bU, 0x3eb6d9a63ee9c14eU,
  0x3e5badedbdddc28eU, 0xbe86fd923edecac5U, 0xbe3a1567bf05aa95U, 0xbe43e133be94ff61U,
  0x3d0c4214bee19f55U, 0xbf21e996bea27f77U, 0x3b878b63be564c99U, 0x3d325279bef226b9U,
  0x3e3c726b3e215b13U, 0x3ddcdbf5bf147f63U, 0xbf007a293d15e9f7U, 0xbf016e51beca85c6U,
  0x3e6ddc7ebf2017cdU, 0x3c3f94543ea362e6U, 0xbe8b9a0abedaf8cdU, 0xbf0a39acbf17e25eU,
  0xbdd5dc01bf004892U, 0xbd48c064bdb312b2U, 0x3e718de2bee07c80U, 0xbf13c871bee8abfcU,
  0xbdac5d1e3e518e62U, 0xbef04bc9bf0bfc4eU, 0xbee9ca1bbf1173bdU, 0x3e478b0a3e430720U,
  0xbf11f7f03b20fbdcU, 0xbedb850bbea6fab2U, 0x3eec3342bbe897beU, 0xbeda4218bde1b5f0U,
  0xbf105475bef28c00U, 0xbe4064743d88526aU, 0x3cc1ee83bd9b34ecU, 0x3ea0366cbece5113U,
  0xbf0e70a3bedbe531U, 0xbf2b7fa33c2378b5U, 0xbf0a660a3e1bc212U, 0xbe4eb2d2bf14e04eU,
  0x3d9e3268be72a98cU, 0xbe446b57be860507U, 0x3c2a4f973c50b4a5U, 0x3e9041c0bf272fcdU,
  0xbd2cf9473e54a782U, 0x3e62a8b8bf03907eU, 0xbf1a798f3eb43816U, 0xbf45dcf2bd316304U,
  0x3c7898b6bea220d5U, 0x3e2595c83d0eecfaU, 0x3cd85556bea1bfb7U, 0xbeb84b0cbf5a2cacU,
  0xbd7c7466bdcdec3eU, 0x3f1211f3bda96e3bU, 0x3d15b5e8bf0f239aU, 0xbea4c933be34d417U,
  0x3db771b23c577935U, 0xbeeccca7bf4a903eU, 0xbe4519cebe6ed093U, 0xbc9e677dbc9ead73U,
  0xbe06c8de3e35cd17U, 0xbdb2aebdbdf8a16aU, 0x3d833a7ebccaa493U, 0xbb49b10a3e8f99b1U,
  0xbe27bb1dbef62682U, 0x3da1cb4bbdce8c30U, 0x3e003e8d3e1d0c6eU, 0x3e814abf3d17c532U,
  0xbdffe6aabd92ffceU, 0xbf09ac5c3e40846fU, 0xbf4681e73f10d921U, 0x3eaaab31bf0fc1f3U,
  0xbf8755e93ea1d251U, 0xbf698ebfbd188256U, 0xbed122e43ebd86d0U, 0xbde8ba8a3e40400cU,
  0xbeafe658bdd583bcU, 0x3f8bd2ba3f09a2ccU, 0x3e4b9d9a3eb617ddU, 0x3e0afee8bc5082e9U,
  0xbedd818e3f17d94bU, 0xbcaac0ed3f8e6649U, 0xbf5de6e4beb2e748U, 0xbefe36ca3dfa7e65U,
  0xbdf7c8033be14e4bU, 0x3e372dfcbf811f49U, 0xbe24a4f13e2b3188U, 0x3f110190bf4e4b25U,
  0xbe147beabea98284U, 0xbc5638f53d6ec55aU, 0xbd3bc89dbf2e635cU, 0xbf22367bbeecaf73U,
  0xbfe1b4463f16a359U, 0xbd85f988bfc7f73eU, 0x3f5c6ac7bedf8e46U, 0xbccf74cd3ed0b59fU,
  0x3db16c1dbd209ab2U, 0x3f48b705bee60b12U, 0xbe89be4e3f01ff24U, 0xbf2e6b913f8124baU,
  0xc001d853bc9e7561U, 0x3f760210bd2a832dU, 0x3df97dc1bdb50aeaU, 0x3f01a9043f8c82afU,
  0xbf7836bbbf072cc2U, 0xbf0510acbef8fda9U, 0xbecbfb8e3dd89e80U, 0xbd0903ea3df0fc74U,
  0xbf3a907dbe817d19U, 0xbd9f705fbf4788b2U, 0x3dd4902a3ed5ad85U, 0x3cf4960abc50fb57U,
  0xbf0fee74bf209b10U, 0xbe0fb8adbbd95c6eU, 0xbf2e57b7be76003eU, 0x3cd6e7203d746d33U,
  0xbe66806dbd97fbe8U, 0x3ee9cb9cbe986ee2U, 0xbea49817bd350b05U, 0xbf1efe79be66781cU,
  0xbec9b0edbef1b60cU, 0x3caff4ba3dd80625U, 0xbf1cc9bfbeb9e753U, 0xbf543109bf15d045U,
  0xbfb6e1873e36eb0fU, 0xbdf7e00cbf244ce0U, 0x3c9077edbf5a556aU, 0xbbd9c8163f09126cU,
  0x3e4313513d773ebcU, 0x3ea4a269bf95ec24U, 0xbebd36503e11cce4U, 0xbf4df25abe90440fU,
  0xbf382a44bbf92c7aU, 0xbf6b8f16be703b22U, 0x3e079eeabd071babU, 0x3ea75571bf4e68faU,
  0x3f1e94273f147055U, 0x3efcb2763f2aebe4U, 0xbc6c9a0b3f43341cU, 0xbdc61faf3f05104bU,
  0xbd154736be60be6aU, 0x3f0c73fe3f257500U, 0x3f02b6983e0101edU, 0x3ef5dc49bd5ef6bfU,
  0xbd8b9df03fac923cU, 0xbd47c12c3f320101U, 0x3eede2f03ec53b2cU, 0x3f1945e43ee678b8U,
  0xbd6487ebbdcc59b9U, 0xbdc414293e0903c8U, 0xbd0458373ed35cb0U, 0x3f51d60f3c8c0905U,
  0xbde75ad7bc984e28U, 0x3ed593573f033b00U, 0x3f80251a3eb33d02U, 0xbc8cb729bcb951a7U,
  0x3e3110e93cbc2100U, 0xbd9a6c6c3e9e2ba3U, 0x3f5f57a4bd84a124U, 0x3c9cd54dba6a0d33U,
  0x3c6a8ac03ec02132U, 0x3ef59f7a3cd4c97bU, 0x3f2f37723d27e11dU, 0x3deddec53fbccf01U,
  0x3e481405bde55506U, 0x3f8922cdbd511995U, 0x3ef9aa93bd8b5549U, 0xbdbddb533f996c35U,
  0x3c0ec0dabd5540ccU, 0xbe2bd83ebd52ec9dU, 0x3eb74a38bf6746ceU, 0x3ce3ab66bf28d82eU,
  0x3eb478173dab96fdU, 0xbf0421583d9d7da0U, 0xbf354d7ebe187986U, 0xbf39a0893d4cb3e1U,
  0x3e6b64383e3559c0U, 0x3d85c43cbf40365aU, 0xbeb212a3bf47cdf7U, 0xbf05ddbfbf4263deU,
  0x3cc567bf3d87a057U, 0xbe15e5ecbef91ed3U, 0x3d9189c7bf016f04U, 0xbdfe22eebe3ca386U,
  0x3e1b92213e9391ddU, 0xbf021e85bf4cd970U, 0xbe9c7dbdbde7534bU, 0x3d3cb08e3e46bd47U,
  0xbbe154adbebdf79fU, 0x3d47eb82bee79f3dU, 0xbe89954d3ec3bbd3U, 0x3d433bf7be833081U,
  0xbdf02744beac62b3U, 0xbf2ce3463eaa3019U, 0xbe9a98a1beef26cdU, 0x3e6d2cd43d604539U,
  0xbe00307d3cd11ca5U, 0xbeb86a173cb75d3bU, 0xbf127c273d5f5c39U, 0xbed5a707bccd6c75U,
  0xbe5fe7bf3ec15479U, 0x3ddab5503e6008b5U, 0xbc96e5d13c8ad5a6U, 0xbdea78263d776d7dU,
  0xbc31c92fbdda3091U, 0x3d5c53773f32b71bU, 0x3e0ee072be6ac8abU, 0x3d0d7226bf513380U,
  0x3ce62aa73f1a570fU, 0xbe6fb57f3d50512dU, 0x3e2991f7be0f099dU, 0x3dce3ab73b395608U,
  0x3e161f9fbecf12faU, 0xbf41370c3db94413U, 0xbe23c9c63d84e263U, 0x3ed177d63e45a428U,
  0xbe1aecef3c9cdb57U, 0x3e2302073d9d149dU, 0x3ed8b5843e2f0052U, 0x3d1dfc383dd05cacU,
  0x3d90ae19bdb184bbU, 0xbf3958a73e99c2bbU, 0x3d6e12a63d1a976bU, 0xbf0440d3bef3bf9cU,
  0xbe84f1f23d09920aU, 0xbd8036adbd78ca02U, 0x3e833c55bd88608dU, 0xbe357dbd3e31a568U,
  0x3e6fea99bee71847U, 0x3f97e091bcf0f477U, 0x3e0ee558bf5333a8U, 0xbeaa8dc03f04722eU,
  0x3dbbc7cfbeb6e1ccU, 0xbea70576bf139ab6U, 0xbf0466a13b90ded1U, 0xbf242d073d93d7e0U,
  0xbf0c9f8cbf5c85d6U, 0xbf2cc71bbf12b8cfU, 0x3e0a4dd2be495ef7U, 0x3d22e2243c9ffb52U,
  0xbf0d4c7dbec69395U, 0xbf0b1431bf2213baU, 0xbe3dad113cf8fe6aU, 0xbe4947ae3d42c383U,
  0xbf09d5383e0d0211U, 0xbebe2f38baca522bU, 0xbf225e3a3d8f95c2U, 0xbf519836bc716465U,
  0xbe7723e9bf1b1e8eU, 0x3d03712c3e19b115U, 0xbedcb26bbe1ca749U, 0xbe96fa50bef22ee9U,
  0xbeb4961bbde67e92U, 0x3dc56e28be879b22U, 0xbf2180cabe37fcfbU, 0x3cbe74d4be2cdd2dU,
  0x3e078c263dc27cccU, 0x3c3e9d12beb255abU, 0xbe4af62fbea0e1deU, 0xbf1dcee4bf44946aU,
  0xbd0946a33d9cef59U, 0xbf4f3705becb2ad1U, 0x3e4706c5bf4ead2fU, 0xbda0d391bf30e391U,
  0xbeb67b303e733398U, 0x3e6fe3823e9c684cU, 0xbed8ab9b3dd74119U, 0xbe9ae9073e87f22eU,
  0xbf0d15e7bf2ff4f2U, 0xbda8085c3f1d2bb9U, 0x3eb90069bec8e18aU, 0x3eb7838b3cf5058cU,
  0xbf011c013ecfa01eU, 0xbe0d1330be174b8fU, 0xbe5b4245bdf50319U, 0xbcaf583a3ed8006eU,
  0xbcf25c123de8c4dbU, 0xbf48404abf135416U, 0xbe382b3d3e7dc40cU, 0x3ee11839be3746baU,
  0xbe260fcfbf02c18cU, 0x3f0bce283f19cb5dU, 0x3f00157e3c3429dbU, 0x3da06d8bbd0b0ec9U,
  0xbe291de0bbd8750cU, 0x3d792bc9bd7203deU, 0xbe7a5428beb6aa5cU, 0x3d28fdcbbee867b5U,
  0x3d8e14623ec81d60U, 0x3d28e158be2ea25eU, 0xbbd35c0bbece2e39U, 0xbf0d8b2cbdd13844U,
  0xbe22803c3d4b9abcU, 0x3ef01df2bd204dc3U, 0x3f12f40abf3ac3ddU, 0xbe91fbb53f4c3e63U,
  0xbf074269be9e2a14U, 0xbe32eefebf17579fU, 0x3e0ea8463d0fb6d4U, 0xbdc0bdd0bd6b4288U,
  0x3d717c90bdcc397fU, 0xbec48513be8845c1U, 0x3e0192c0be9f7307U, 0xbd985a3d3ef3f4f0U,
  0x3de5f0aebed161b7U, 0xbe392098beefa746U, 0xbef06c2ebebc0d9aU, 0xbea589983dadb41bU,
  0x3b2555be3ef80192U, 0xbf059761becaed67U, 0xbdd430413adf576eU, 0xbf02562abea4a3c7U,
  0x3ce392b73d6e1c37U, 0x3d1a32aa3d0c655cU, 0xbf1218b2be778c35U, 0xbd200e30bc02a68fU,
  0xbd11afc4bf03b2e2U, 0x3eef0124be880822U, 0xbf19f6aa3dc48917U, 0x3f00b4c0bed3e019U,
  0x3f335e59bd1a5a3aU, 0xbd0215673d8e66a6U, 0xbe81f3febebe2966U, 0xbf09f3a7bf0705b8U,
  0xbe688dc13f052d77U, 0xbf0e9f7d3ce68ed4U, 0x3dafa32abe24d69fU, 0xbed30151bf12df56U,
  0xbe976189bd74770eU, 0xbecf706dbf0c0ad8U, 0x3df883c2bf88b2cbU, 0x3e0e66adbf23e437U,
  0x3d821cb73df83778U, 0xbf5ededbbe2471d6U, 0xbf0a4d5bbb938341U, 0xbf0e3bff3dafac9bU,
  0x3df5ed83bee88487U, 0x3e56a773bf2d4cceU, 0xbeb8b90ebf02d34fU, 0xbf0b0fc4bf349139U,
  0x3d2038a13b1c9164U, 0x3d134809bf16ca55U, 0x3e14a74bbf290893U, 0xbe848a7dbf0700f2U,
  0x3e182ad63d848c4cU, 0xbe69c3edbf2a1debU, 0xbf01862cbeee401eU, 0x3da2082e3acfcfd7U,
  0xbf0961efbea17d08U, 0xbca2311dbf2568e8U, 0xbf52c4173d333a60U, 0x3bdb19f83ddc3392U,
  0xbe0b46d2bebc6d8dU, 0xbebe55ddbcb15330U, 0xbf373e19be2fb3e3U, 0xbd654c45bdb779ecU,
  0xbea38108bd177aedU, 0xbeec1ceb3e34c742U, 0xbf0babe13dee40d0U, 0xbe517150be607662U,
  0xbf85e65f3eac1390U, 0xbf95537fbd788e81U, 0xbed23af23f4ba5f7U, 0x3d1001633f1d7717U,
  0xbeab0aecbda04d25U, 0x3ede3cc83f474898U, 0x3ede174d3e900f5fU, 0x3ee38385bb983518U,
  0xbe9bc3803f21b10bU, 0xbcd4621e3e95eb12U, 0xbf0c9d8bbf8c3316U, 0xbeee545d3f0297f2U,
  0xbe17bcb93cf45f2dU, 0x378dc66dbfaef40dU, 0xbd9596723f0c487dU, 0x3f302fb8bf13fb44U,
  0xbd40f190bdf8744dU, 0x3d8c30bf3f0c33c0U, 0x3c01737fbf431b08U, 0xbec34b92be96befbU,
  0xbfc423143ee37500U, 0x3ccc5c8ebf3e1f85U, 0x3e8211fcbe90e1ceU, 0xbdc4b6763e8b80c8U,
  0x3d0135df3e845627U, 0x3f199c95bee139ffU, 0xbea359053eae5ca7U, 0xbf7b9a0f3f1c9714U,
  0xbfa673b03cb7e7dcU, 0x3f8e2973bc2bef39U, 0x3ede220dbc5fde55U, 0x3ee7cd053f3ed2dcU,
  0xbe8735673e4b8843U, 0x3e1835fe3e5cc704U, 0xbe63b0d33dfcb993U, 0xbe76524a3ecb89daU,
  0x3cf56bdabf03be01U, 0xbddfe2e73f49cc55U, 0x3f0d9b1bbeaa0177U, 0x3e1103783d7e34abU,
  0x3d6067103f47bc65U, 0xbec5bc8fbdf88209U, 0xbe3d69e33ebbbfe4U, 0x3ed37ead3ef2e468U,
  0x3c05854b3d569f04U, 0xbf7828c93ca889a7U, 0xbe40a3103e167547U, 0x3f171e743eb4cd17U,
  0xbdf4f876bd786f98U, 0x3f5b89423f11259aU, 0x3ec79928bd2c3586U, 0xbc4a4e9f3d097af1U,
  0xbdbaf73abde7bc84U, 0x3c97afddbe2c6ef2U, 0xbdcd6e54bcef418eU, 0x3d436d58bee789f6U,
  0x3da5bc473f1def90U, 0x3dcae9633c11b354U, 0x3d9c25a4bde2a58eU, 0xbf03c87fbcdd7a44U,
  0x3e3b9d3bbc5a8385U, 0x3f6cc902be1e23ceU, 0x3f227df7bf08ff42U, 0xbe3769673f2e0d96U,
  0x3dc2c4483ed6bce7U, 0x3eab87453ec38470U, 0xbbcb43bbbc168700U, 0xbe7d5785398e7beeU,
  0x3d2852633bfa72afU, 0x3cb2aacc3f80fe8fU, 0x3dcfc2b1be88acccU, 0x3d363eecbf2ef4bbU,
  0x3d0464253f4a84afU, 0xbd83611b3d8c0e8fU, 0x3d2498533e60b6d9U, 0x3eb298143dda5d9eU,
  0xbe48b2b3becab9edU, 0xbf3415b1bcfc0ea9U, 0x3dd33bbdbc6353cdU, 0x3f1cbc313e4976c4U,
  0xbe1c21623c48564eU, 0x3d7cb68ebe12de40U, 0x3eb26cea3eaaae35U, 0xbc1aae0a3d44a63cU,
  0x3e588dbabd4877ebU, 0xbf3e224cbc714eb1U, 0x3d55e87b3cfd6b1dU, 0xbec9414dbf1f4be6U,
  0xbefe35293d364995U, 0xbe13ed0a3d24538eU, 0x3e4fe864be0a885cU, 0xbea51ddf3da179efU,
  0x3d15b2a4becaf9f2U, 0x3f8b3134be66f9b2U, 0x3c4faab2bf098037U, 0xbe78539e3f55d112U,
  0xbe63fb203ee00953U, 0xbe0838ecbd4c245cU, 0xbbb11de9bdc212d8U, 0xbce4e3f43dd9f470U,
  0xbd665c19bdc8b81bU, 0xbe7cf0d13e81e0b1U, 0x3cbb6fafbf715186U, 0x3d260b633e0224c0U,
  0xbe8e12493e0cd9e5U, 0xbaee6c31be71dd2aU, 0xbea135b1be06b60fU, 0x3c58c4353d23c61cU,
  0x3c7446813f2783f0U, 0xbeef3f59be2cba6aU, 0xbd2323cd3ad7262aU, 0x3e31fb13be81acdeU,
  0x3e2ebe86bda0c077U, 0x3ee79fd93e4a3a69U, 0x3e99df973e7bcf2aU, 0x3e4209273e292e10U,
  0xbd786da7bf423fa5U, 0x3e5845c9bd5cc53cU, 0xbebb1f88bdfa12baU, 0x3ea6f805bf1908e2U,
  0x3f0255083ec7551cU, 0xbf12b3a33df735e4U, 0x3db10f78bf1a377eU, 0xbf37389a3dc08c59U,
  0xbe66940a3f25e0f4U, 0xbb056592bda8e638U, 0x3ea7971abc383675U, 0xbf4432823ec9d89bU,
  0xbd88d1bb3ea85c79U, 0x3e0c4f7c3e27cecaU, 0x3d69d058bb243be9U, 0xbd815b193c94ae58U,
  0xbdafbdf3be111b5dU, 0xbd04e2963f385858U, 0x3d9dde7dbe894bd7U, 0x3d5b4b13bf350fbcU,
  0xbb817cc93efc3295U, 0xbe29b64d3ca74eecU, 0x3dfec219bd03f505U, 0x3ed40949bc4825e7U,
  0xbd984b54bee3d699U, 0xbf046479bdc746fcU, 0x3b0669e73b4c9acaU, 0x3ef80bd93e63ee82U,
  0x3c5506b83c4e594fU, 0x3e31429f3cdb8f44U, 0x3e7a01693d63e36cU, 0x3d03da483d17a5a8U,
  0x3e473f43be3116fcU, 0xbeffbe2e3d859a50U, 0x3cb1d6b13d549bfdU, 0xbef739dbbee6500bU,
  0xbf07639e3c6f67b2U, 0xbe0900cd3b144211U, 0x3d4cb4afbcc0874dU, 0xbe9406443e770263U,
  0x3e136725bef5c51dU, 0x3f81a7a7bd5542e5U, 0x3dc886c5bc2ae816U, 0xbe8c82573f61826aU,
  0x3ed3c87f3e6d140eU, 0x3e14cb773f0deb67U, 0x3ebed197bc9cefafU, 0x3d94b0b0bca4488aU,
  0x3f1402a93e1c175cU, 0xbd03f17f3f3740f7U, 0xbd168ec6bc63cbccU, 0x39abafd8be07ac37U,
  0x3efccbc93f01ffb8U, 0xbcdf978b3d29f3edU, 0x3f1b3651bd19117bU, 0x3d33d2bebdc403b8U,
  0x3c491e3ebdbc2322U, 0xbe17b9c43d26d42eU, 0x3dc9b1b53d46a3c1U, 0x3f54f540be91a1c7U,
  0x3c149ce73eafceb8U, 0x3dc8319fbd0f2c62U, 0x3f1324323c5e6c6cU, 0x3e7ef98b3ea34695U,
  0x3f423873bd8a0febU, 0xbd916cde3e91d3cfU, 0xbd010ff83ee5b996U, 0xbd8e3c51be1a054bU,
  0xbe56b995bd1a8750U, 0xbde3e7563f179f14U, 0x3f028956bd0bf517U, 0x3ed94292bd4f22a9U,
  0x3e62c1debe3e9133U, 0x3faf0d8fbcd55209U, 0xbdb0d973bd353852U, 0xbe17feca3f60d1a5U,
  0xbeb6948b3e996c62U, 0xbd2d3ab23e999c15U, 0x3bb807bc3e179718U, 0xbe0c102e3f035ee7U,
  0xbe64075bbe8841b6U, 0xbe3d17183f32585fU, 0x3f0a4e00bed037bdU, 0x3f23b465bcd8360dU,
  0xbd1549403f23789aU, 0xbe496f33bd4524cfU, 0xbd6b02b73ef7ef74U, 0x3eb67cd43f371535U,
  0xbcf218693d2cba40U, 0xbf4b74453adade48U, 0xbe1a2a6e3eeed8a8U, 0x3f16c3ae3e687e0aU,
  0x3ccd6c4d3c34a1a1U, 0x3f252adf3f463c2fU, 0x3eaf355dbe9fbcf8U, 0x3c8e42393b9cce2eU,
  0x3e2b7483bdf9f049U, 0xbca47b5b3e4f75abU, 0xbe0d9966bc9d8542U, 0x3cb098fcbefe1fa5U,
  0x3d684dce3ee540a8U, 0x3d9652a83c9f34e6U, 0xbe4bf23dbe36050dU, 0xbe97153b3d3298deU,
  0x3dbc828e3d8464faU, 0x3f7f6cc3bce81a6dU, 0x3f2c3d81bedd05efU, 0xbe64aee93f85ee65U,
  0xbee444e4be02320aU, 0xbee23e2fbf004ba7U, 0x3e0879ddbd65ed07U, 0xbc0a3f7d3cc4117aU,
  0x3b84dbf9bdc4dbbbU, 0xbf1702b5bee0bf73U, 0x3c860020bf3d8b5fU, 0x3d6822013e8d4680U,
  0x3d5b1467be91bb6bU, 0xbb755da0beba5e22U, 0xbe3bd24fbeb513f6U, 0xbe05092abc4a59b4U,
  0x3d81f2ae3f110775U, 0xbed624d3be96b90eU, 0x3c0574953ccdb9f5U, 0xbe88148fbee5d7ecU,
  0x3d7169c03d409feaU, 0x3d9043733e286a4eU, 0xbead1cb6be8d0a53U, 0x3b58f9ff3de5ec3aU,
  0xbe3746cfbf70cae8U, 0x3ec44a65be730aefU, 0xbf13f3253d5febe2U, 0x3ec9c093bf2d4b72U,
  0x3ecc73ec3c2a3f6eU, 0xbeede2fa3e1c9f6cU, 0xbed1b055bf887137U, 0xbeda5338beda26ecU,
  0xbe43632c3efe084aU, 0xbea947e73e03857eU, 0x3ce7f2d1bd935b6dU, 0xbf8ef0bdbebbbd3cU,
  0x3eb609d73e8c07a2U, 0x3ea443ca3e9373cbU, 0x3f05c9e8bd82374bU, 0x3ccd3e84be427148U,
  0x3edee05abd59411eU, 0xbceaf17d3f5bbcc5U, 0x3bf2f881bda3ef04U, 0xbd2c205bbdf2bbd5U,
  0x3ea7068f3f2b2f87U, 0x3d253de8bcb0c148U, 0x3ee9f9eabd8c8f69U, 0x3ed5c440bd0a153dU,
  0x3cd36ba0bc3a6545U, 0xbd9fbd15bd01ef57U, 0x3df386613b2b3daaU, 0x3f1649bebe27cb0eU,
  0xbcc491123ed4e51aU, 0x3def5196be07deb2U, 0x3ec3f87d3cfe8521U, 0x3e67d5c23e7f9bc9U,
  0x3f0a6a4fbd2b7be7U, 0xbe06e3fb3e923decU, 0x3caf9c473ec96a8bU, 0xbdd8fe2dbdf48632U,
  0xbe0f8c3f3d845ebbU, 0xbe301d0b3f15d151U, 0x3f1454cdbdd56945U, 0x3ebd97d23d3f9a04U,
  0xbd1d7475bcf80890U, 0x3f6ae4cbbcc62a4fU, 0xbda5a3693c5b02d0U, 0xbe06ea723f6080efU,
  0x3f29db23bd54d3f2U, 0x3ba5e0b13d7e4d37U, 0x3e9c2e5fbeaffa09U, 0x3e140df4bf3648b4U,
  0x3eaad3693d7660f2U, 0xbdb8a8c4bf0abbccU, 0xbf165702bb274866U, 0xbf39961a3d8536a9U,
  0x3e2bf3783e19960eU, 0x3de3bf30bd962ebaU, 0x3ee727acbea52506U, 0xbe10e0c2bf43edbbU,
  0x3df1966ebd733313U, 0xbd51caf5be71f105U, 0x3e0542a7bf2b10a1U, 0xbe43a709be090799U,
  0x3e35efa23ec9faf5U, 0xbeffd237bf446c55U, 0xbdd08906bd7d973aU, 0x3e03854d3dcd5cddU,
  0x3d27253bbd4f170aU, 0xbb43b6083da26263U, 0xbbcce2773e822800U, 0x3c4dad1abd8890aeU,
  0x3ca861b3beaf06ebU, 0xbe3dc4453eca4f60U, 0x3f0a7cfebdc0771cU, 0x3f16e902bca4da79U,
  0xbe12584f3da327f3U, 0xbe7dabb23cf061e6U, 0xbee6d7b73e3e57d2U, 0xbd8e4f20bd8f911eU,
  0xbec64cbcbea4567cU, 0xbe9439dbbf348e60U, 0x3da13162bdd85a7eU, 0xbb205173bb28ec0aU,
  0x3d49bca03b16e9a1U, 0xbf1f6f17bead1f93U, 0x3ce38104bf31cee7U, 0x3c7ff0cf3ede69caU,
  0x3dddeaf8beacf888U, 0x3d4b0970beec5b60U, 0xbf176bc9bd93973eU, 0xbedf87113d9a569bU,
  0x3d4c081c3f1e9a8bU, 0xbedcf6f0bf2d2597U, 0xbca21f71bc9bba5aU, 0xbf08bf67bea45f6eU,
  0x3e0100873c4e30f1U, 0xbca39d3e3e3aa9edU, 0xbe316387be7f6539U, 0x3dc7ebea3d403eccU,
  0xbe105a91bf6eb49dU, 0x3eacf4efbf07ba30U, 0xbf2df1153ddc3641U, 0x3f078820bef45158U,
  0x3f1743d63de34135U, 0xbef651443d913e9eU, 0xbe8fbe89bf4707feU, 0xbeb1f811bf21bd5fU,
  0xbe5f9c083efa53efU, 0xbf2198c43d6c57a7U, 0x3d3d5c5abd89b541U, 0xbf5c785bbf3a1e38U,
  0xbe5d013a3f23c55dU, 0x3e93e5053e570dfeU, 0xbcfd359cbdaa9b1aU, 0xbd9d1d88bb10110eU,
  0xbdba5758be296e59U, 0xbdd707713f7737bcU, 0xbbd75a07bf14a53aU, 0x3d4ad2b43dca8256U,
  0xbd05529d3f2e7e7fU, 0xbddac53cbd4f5e8aU, 0xbe0624e83e8cd46eU, 0x3db34aaf3e39ffc8U,
  0xbd5a9b723e5f59c8U, 0xbf349181bd1ceb0cU, 0xbbd2ab053d91bf12U, 0x3e901b123e6b1284U,
  0xbdb1fef03bd123fcU, 0x3f54e3753e1e75d6U, 0x3ef348bc3e33f225U, 0xbde08e6e3ddab09dU,
  0xbda08418bebdb2f6U, 0x3e1663113e4bc7a9U, 0xbdfac6903e054303U, 0x3e128d22bf0fd2ddU,
  0x3e5c1f953eea9787U, 0xbe1fb42f3d12efe8U, 0x3dd44574bf0e329dU, 0xbf1433cd3d8be071U,
  0x3e969e9e3e7afd47U, 0x3f505c73be2de610U, 0x3ec894bfbd2ff162U, 0xbf1675c53f8fb0ecU,
  0x3f2cf0683e44f631U, 0x3ebc49803ed47b24U, 0xbd19d7a13d6b1b0bU, 0xbdd943e13caf2e13U,
  0x3c9845a73b3fc90eU, 0x3f0c70ee3f279132U, 0xbc8dadac3d24fb84U, 0x3bc56b20bf3ee1fbU,
  0xbc468c653f2a1741U, 0xbdb98b163f0fbab1U, 0x3e88b1133e84fe11U, 0x3ecb387d3cf43897U,
  0xbd93e7a9bf0db5e6U, 0xbd0dc2853f29dae1U, 0xbd0d312ebb0c2a5eU, 0x3f0028103d25d771U,
  0xbd108803bde7b555U, 0x3c92ab49bd49a196U, 0x3e68e7db3ed0d8daU, 0x3bba1d79bba3052eU,
  0x3e072fb13e0a8394U, 0xbf1e782a3e7a72a5U, 0x3f43e4e23d2eb852U, 0xbf2c41c33c42e30fU,
  0xbec66d64bd490dafU, 0x3e013ee63d54a7e8U, 0x3ed842893ea4fda2U, 0x3f3067ad3f43fd3aU,
  0x3e94c729beecd5dbU, 0x3f4c0dc6bdb54a73U, 0xbde7cebebcda1cd4U, 0x3dcacb6f3f0cf242U,
  0x3e1adfdf3ef38b1cU, 0x3eab62fe3e199d75U, 0x3b865fde3c8043f0U, 0xbe10c0813d356facU,
  0xbd85301dbe4cb2dcU, 0x3d3d5fbc3f81fe35U, 0x3d623cfbbe7607b1U, 0xbd0b7ea6bf3dcb56U,
  0x3d77ea0c3f183d52U, 0xbb6f76503d0458e1U, 0xbdf74c4f3e2b734aU, 0x3ed3066a3d87ab46U,
  0x3ca7deacbef58cebU, 0xbf3093863e1bfdb7U, 0x3de73d5a3db2765bU, 0x3f0e2d91bd23b904U,
  0x3c1bca373d894e42U, 0x3e322afe3cd5250eU, 0x3ead5ae23e19f990U, 0x3dc0f93dbd18db45U,
  0xbb03731ebde1d2f9U, 0xbf01e9c63e546162U, 0x3d05b3233db040caU, 0xbf0bcfc4be55864eU,
  0xbef42c6e3dd17e23U, 0xbd83a0333ca5e56cU, 0x3de638acbdd7f579U, 0xbe8daf5c3e7c9cb8U,
  0x3e1f0ffdbf0a66fbU, 0x3f7aadcabe94832eU, 0x3dba0a1dbebf7b1aU, 0xbe773ec13f4fb5d7U,
  0xbf279cebbeb0c8d9U, 0xbeace487bf33754bU, 0x3dedfbbabb81fdd8U, 0x3d8e8db63d93379aU,
  0x3c07c5293da8212aU, 0xbefbe697bf0853dfU, 0x3d137569bf35a1beU, 0xbd0c89483ec2df61U,
  0x3dcf4a2dbef34e7eU, 0x3a5d4bb0bf08ed3dU, 0xbe836c67be9ccadbU, 0xbe1d1075bc2b0770U,
  0x3d4aa6593f2eab88U, 0xbee417dabec09775U, 0xbe4a50a9bd8be453U, 0xbf445755be68c13aU,
  0x3e0135de3cff7692U, 0x3d22ae813ceff259U, 0xbf0870bcbe344f93U, 0x3d9d8397bc79266eU,
  0xbdf47b13bf0db5e7U, 0x3ee13334bf0b88c3U, 0xbf0ffe903d9e8e6cU, 0x3f058d69bea55b26U,
  0x3f2baadbbcb89d7aU, 0xbdbd222a3d9f962dU, 0xbe4fe3c9bf1aca49U, 0xbe8b57b8bf189b75U,
  0xbe76626c3f07caa1U, 0xbf0ce93fbd937e32U, 0x3df061e9be4d623eU, 0xbf109288bf0a10a4U,
  0xbf92cfc3bea2b91aU, 0xbf04d0b6bee771f5U, 0xbdc043973cdec373U, 0x3e50b8f03d94fb60U,
  0x3ce30336be97e6baU, 0xbd623697bee23541U, 0x3d7176f23e610a56U, 0xbd560f813ebe6c28U,
  0xbd9c5cc4becd450aU, 0xbe18bdfdbd5f9569U, 0xbf2270f1be9efaf3U, 0xbf12835c3c21e238U,
  0x3dc70a2b3e855f08U, 0x3f2b36a2befeda1eU, 0xbd7d99a8bcff78e8U, 0xbeeca9bbbe9336c6U,
  0xbc57a5703d91bd1fU, 0xbd93e2c03d8d0a35U, 0xbea861f8bf1d6609U, 0xbec8347a3d52b4d2U,
  0xbe8039213e6d6842U, 0x3e9a733abf3b85adU, 0xbbccab753d923913U, 0x3e906dc83f017b2eU,
  0x3ed166193c052af0U, 0x3ea141d3be37f6d6U, 0xbee880163e7cf6d0U, 0xbf4c4455be53454fU,
  0xbf0667d83ebd29aaU, 0xbea97b48be97cee7U, 0x3ca6df933dd779edU, 0x3f08d024bf18c80bU,
  0xbef5c1afbe6d40bcU, 0xbd82b764be131f1eU, 0xbf00cbfa3cd66830U, 0x3e4fc6f4befd0ba6U,
  0xbf0a26933d022f74U, 0x3e393ba9bf1122bdU, 0xbf0237d13defb1fcU, 0xbf10cb3d3c95be22U,
  0xbf054441be9b93c7U, 0x3e93338d3e5bf874U, 0xbea0955bbe7ab243U, 0x3e4856afbf0056a4U,
  0x3dc0161f3d519e40U, 0x3e52bbbcbdd181e5U, 0x3d9cba1abe8c6d37U, 0xbe5e855c3e495d9aU,
  0x3dc4a9b3beed05b8U, 0xbef37b19bf25264cU, 0xbe818679be17354fU, 0xbea5383abeb503b3U,
  0xbf9ba80e3e6b6ad2U, 0x3cb0d2c13d9b745fU, 0x3e3b6e0ebedabf3fU, 0xbc2a1d0c3e6ae66fU,
  0xbe734e95bf242dc9U, 0x3da26302bf83ec60U, 0xbebee6353e5a5b46U, 0xbf0bae1b3d4bafc1U,
  0xbf11ce513d80afb5U, 0xbf14c2263bbe2d9bU, 0xbf3a3c653e7581e7U, 0x3e6d4488bec944b2U,
  0xbf0e5c743e8c2957U, 0xbd3775ad3eefed32U, 0xbcf25a593d3f2df8U, 0xbe49c5bf3ed38966U,
  0xbdadd187be54233aU, 0xbdd9d44b3f6684a2U, 0x3ee11590bed903daU, 0x3ee22fb23c88315cU,
  0x3c1f45353f1fcca0U, 0xbe70197fbe4e518cU, 0xbdbc45ab3d4128ffU, 0x3ef6eeb83ed8e50dU,
  0xbe11e477bd6f4138U, 0xbf89f48bbe065bddU, 0xbe36a8ef3e09a2baU, 0x3e68babe3e861ce9U,
  0xbd18b3f3bd63ece5U, 0x3f18d6d53f0e18daU, 0x3e3570e73ef9cd44U, 0x3cb6bb96bd9669cbU,
  0xbe969b7bbe04d759U, 0x3d86c0efbbd5f7ebU, 0xbe312b303c3a6498U, 0x3bc0f940beea9067U,
  0x3e3a801c3ee73523U, 0xbd0b1fd53dc80000U, 0xbdf491adbe60e156U, 0xbef493c93dcc86a1U,
  0xbd095e09bd627091U, 0x3f296f06bdb363a3U, 0x3f427cb7bf0e0c63U, 0xbeca52953f07bc68U,
  0x3f64ca673e867af1U, 0x3e34c5dc3f0d9eb4U, 0xbd9216d83c19a417U, 0xbd00598c3d3ef43cU,
  0xbd9108df3d075649U, 0x3f3082e93ea193e0U, 0xbd6248e13dfe6460U, 0xbbf1e454beffd02bU,
  0xbd67bab03edbb777U, 0x3bf0d7073f4229a7U, 0x3ec9cc423d0f0e60U, 0xbd90d496bda15bddU,
  0xbe4ee59fbeca3bceU, 0x3dfe79433f0c2e99U, 0xbd81f2d43cb6e259U, 0x3f0e80eb3e08948fU,
  0x3c9bd191bdb81044U, 0xbbe5d411bc29bb3fU, 0x3e9e58893dfa8792U, 0xbdff1617bd2475f7U,
  0x3b837bb83eb3cf5bU, 0xbed817b73e8112b5U, 0x3f53fedebdaa1e51U, 0xbedf413c3d2366c9U,
  0xbecb6c09bc95cda2U, 0x3d37a68abd91b0edU, 0x3f165cf93ead37c0U, 0x3f3c61523efa11f7U,
  0x3ea554a6bedd7521U, 0x3ef17c30bd6aa1faU, 0x3d3649633c0eaef4U, 0x3e4d80a43ea0e36eU,
  0x3f3287093de55affU, 0xbe82bc563ead23a1U, 0x3e97769abef11b7dU, 0x3cc5784ebefcda6fU,
  0x3e84bd2b3dafc567U, 0xbd0b564f3ee42896U, 0xbedac46abdf80883U, 0xbf0a72e3bc114d4cU,
  0x3ea9d768bea54f08U, 0x3c82478d3d57e026U, 0x3eb968e9be1a24c8U, 0xbdb78bf7bf0fc026U,
  0x3c6029953d571ac1U, 0xbe1e99013e01f67aU, 0x3d177210bef5d288U, 0xbded14dabedd41bfU,
  0x3d79ff903ebb7946U, 0xbea7c98ebf191cc7U, 0xbe352eedbec9df83U, 0x3e9310973e873188U,
  0x3ca5bc41bdba40bbU, 0x3d0e3fc5be7c4280U, 0xbd4086013e8abe73U, 0xbd19ce94bde67c37U,
  0xbdafbfb8befedd62U, 0xbeb592c43ed8c007U, 0x3f473f65be39a1c7U, 0x3f0ce6783cff3f7eU,
  0xbd9c671c3d4216c3U, 0xbd991c963e4cf522U, 0xbedee8543c0c2360U, 0xbe5733ed3e3d628bU,
  0xbe5e932abe4e08b0U, 0xbf12fed7be1a0da3U, 0x3e268acabf61772bU, 0x3e3a0962bf4c60f5U,
  0x3e15e9703e82cfeaU, 0xbf830ad73df7432cU, 0xbf0ecd0bbea12353U, 0xbf12eff63dbac8e8U,
  0xbd0716173d8fa9bbU, 0x3e0b1fc2bed6fe37U, 0xbeeee0d0be93752eU, 0xbed39086bf5541bfU,
  0x3d7498bb3d7f63a7U, 0xbe20f1b2bf22c77dU, 0x3e1dc6abbf3e4d1eU, 0xbeda2f1bbe8b1844U,
  0x3dd4451e3e030158U, 0xbeac89a3bf2c2c61U, 0xbec273f3be50cbadU, 0x3d9b87553da73536U,
  0xbec0b522bec2ad89U, 0x3d033042bf07e60aU, 0xbf50e6af3d7874baU, 0x3cd02932be989371U,
  0xbcb764abbf33eb4cU, 0xbf583d173d9f76c7U, 0xbf63b8afbebeddc0U, 0x3e29a229be710527U,
  0xbec9be1d3b03e391U, 0xbe7c93bc3d99c704U, 0xbf02e0493d1109aaU, 0xbf1cee0abdcfca12U,
  0x3e99561d3ec0d815U, 0x3e6194b83eb4a22cU, 0xbe5b26d53e1e38c7U, 0xbe71568c3d918df3U,
  0xbe93a029be7745c2U, 0x3fa7279f3f53dd77U, 0xbc92f81d3eaa69b2U, 0x3c74810fba32c0e6U,
  0xbe222c3b3ef7ef9dU, 0xbe8d0b5b3f9619b7U, 0x3ecac4543e7e98dfU, 0x3e03b7553dcea405U,
  0xbefbe31abdce48efU, 0x3e82cc463d5b5a32U, 0xbdf8bf6d3ddfcc77U, 0x3f64fb673f06e078U,
  0xbeeca05dbe54b68dU, 0xbd8af7913d0556d1U, 0x3f1373c93e1c100eU, 0xbed1d0debec936ccU,
  0x3e9997943f2544fdU, 0x3d3836d03e678ca2U, 0x3f88924ebe684649U, 0x3dad2c413ed16573U,
  0x3c8c6c7abd8dcbaaU, 0x3f1bbd1abe7cf152U, 0x3e90f80a3f3d837cU, 0xbd608e263f907357U,
  0x3e384dd7bdddd3ccU, 0x3fafe91ebe5a71daU, 0x3d3a650fbdbacdaeU, 0x3efa8bf03f225085U,
  0x3d896f2e3d9ce68bU, 0x3d6166963dbd34d8U, 0xbe865fad3de79897U, 0x3d45e27d3de68ec0U,
  0x3d45cecc3e1b7ec7U, 0x3ba63a0b3d697216U, 0x3d9f9ad7be758955U, 0x3d2a9b23be6536cfU,
  0x3deb9c913d31395dU, 0x3db9272f3d6b4a0bU, 0x3e19829a3ded7147U, 0xbe8bddfa3ccfa4f5U,
  0x3e01fd793e1e88b9U, 0x3bdb251d3d792b5cU, 0x3da9dae13d17ea8bU, 0x3d822cb83da682d0U,
  0x3eda1b313d3d0c6dU, 0xbeaf74383e9d42a0U, 0x3f2fd46abdf92b9dU, 0xbdd016f0bebaa87bU,
  0xbf0a1191bf5b2fbfU, 0x3ebf09a33f1100ffU, 0xbe88ea5f3e4806adU, 0x3d5286503e43569cU,
  0xbf8c1d563ddec091U, 0x3c55171cbd22e0c8U, 0xbd0285d3bf5fae89U, 0x3eb9139c3f3bdc07U,
  0xbcd40673bf4ee920U, 0xbd0854d1be2a9770U, 0xbe09fbcf3eaaf973U, 0x3f084867bf3b79ebU,
  0x3dceca10be88fb03U, 0xbf63d893bebf46c5U, 0x3ee0cc043e6cb464U, 0x3e9cb9f13e7b0804U,
  0xbd865d8cbeab805dU, 0x3e4545913dc2e8a2U, 0x3ebea7443ec34d98U, 0x3cbede123f0b0dc9U,
  0x3d8090093e4f9044U, 0xbd7f44a4beeab84aU, 0x3ebe6e6fbe5da86bU, 0x3f02306f3df90bffU,
  0xbf534822be1f9a41U, 0x3dec74efbf04ce63U, 0x3e8187c93d91055eU, 0x3df78e55be128fd7U,
  0xbfaea2073e986f23U, 0xbf74449bbf72fd26U, 0xbe8f44f83ee1ddc9U, 0xbefe95e3bef016dbU,
  0x3eca24d93e35f154U, 0xbecbada8bfb00379U, 0x3e81294d3e1acfc2U, 0x3e1419253d2f0811U,
  0xbd853939bf252eccU, 0x3ef6cfd03e918fa8U, 0x3ecc0cd53e413012U, 0xbeae6fd1bcfe470fU,
  0x3e2b82b6bd7f5201U, 0xbef97aa7bde0eedfU, 0x3de26fe8b782bfadU, 0xbfa82af53f3b0083U,
  0x3efb48f9bf0ab315U, 0x3ef7e9983f37fd6eU, 0xbed308a3bf6d8907U, 0x3ee6bd513f8371b4U,
  0xbf6610a73f08ac54U, 0x3ea4a22b3f0e523fU, 0x3eb468f5bf4ba48eU, 0xbfa2323abf1c930aU,
  0x3e9997453f12f959U, 0xbf6532e9bf7fd83fU, 0x3eaaceea3ed8de87U, 0xbf38f64bbeef596cU,
  0x3f1c1e963f241216U, 0x3f1508fbbf276e96U, 0xbf671c78be82ddbcU, 0x3f08ba3ebee3a4f5U,
  0x3e660a233ee64b03U, 0x3e59c3c73f5a9020U, 0xbf3e78fcbf47a208U, 0xbe637a3abf158f0fU,
  0xbee6c9f6be61b8e9U, 0xbd25a47b3dde5753U, 0xbfa9a58bbe9a0cd6U, 0xbf90e959bf4cdf7cU,
  0xbf5ef2143e77b3afU, 0xbec768bdbf851bfeU, 0xbfa1863dbe85dbbdU, 0xbf3af7d43ebbc48dU,
  0x3e99a791bee11235U, 0xbd6a2f0a3ef75ee0U, 0xbf8064753e9c551fU, 0x3ec0a323bf6861ceU,
  0xbf865addbdbf7d91U, 0xbf8467ccbf146e90U, 0xbde20642be2f1e19U, 0xbea463d0befc65f7U,
  0x3e78d295bd4d5218U, 0xbf26558dbf719451U, 0x3ca9bca5be1e6a13U, 0x3ca8829ebf0a4423U,
  0x3e6e97c9bf23784cU, 0xbd16c7103e694880U, 0x3e6b38f23eb64ce6U, 0xbede8125bf57948cU,
  0x3e9512d13eee176cU, 0xbf2e00a43d5a5affU, 0xbd4f3f6bbf4f651cU, 0xbf79dbc3be37fa25U,
  0xbfa59d903edbd8ceU, 0xbf732cecbf6a9f8cU, 0xbe128f6c3ead1b23U, 0xbf094046bde3860eU,
  0x3dc160113d871128U, 0xbef343c7bfa0dc93U, 0x3f119021be86ad74U, 0xbd3cd02bbd519e45U,
  0x3efdffc5beca1c9aU, 0xbd42e927bce8bc00U, 0x3da48953be51e569U, 0xbe906cc3bf7242f5U,
  0x3e151c4f3ef2201aU, 0xbf084c06bec5f9d7U, 0x3e3f3f01bee663cbU, 0xbf6fd3d23f00d4e8U,
  0x3dbeaf3a3ec5ca6cU, 0xbf806748beafae3eU, 0xbea5048e3e5431e5U, 0xbf483342bf1cac34U,
  0x3f1e79903e63afbeU, 0xbf4601babf1f4100U, 0xbe3102d9bea16aabU, 0x3e2cfbadbd0e1d1bU,
  0x3e8c99efbf337608U, 0x3e1c7b0f3e7a15eeU, 0xbd821b383de5754dU, 0xbe9327ddbf153816U,
  0x3e3ead1f3e0612cdU, 0xbf68df8b3f3d1ad9U, 0x3e9550f7bd849eecU, 0xbf96e684bdd1b8f7U,
  0xbfb636233e61bb56U, 0xbf249758bf6e183fU, 0x3dc3c89e3ebcd488U, 0xbf008f9ebed774eaU,
  0x3f303c2f3cb98433U, 0xbf2b35e7bf9ca20aU, 0x3e2c4b673e2a86faU, 0x3eaae0d03e789df1U,
  0x3ecf48aabf3ed869U, 0x3f2106503e9b7622U, 0x3df22d863ea9cd76U, 0xbe3067dbbe489e14U,
  0xbe708d3a3e996b6aU, 0xbf19b83abd9c065eU, 0x3f322475be1ea2d4U, 0xbfc7ca443f352854U,
  0x3da9943d3f8d9e54U, 0x3d5c55c53ea1f540U, 0xbd8d4b6f3e2c4855U, 0xbfba234dbf89a4f6U,
  0x3f1bd9073e25e673U, 0xbf9afd81bdd5e261U, 0xbeee3fdfbd2995d4U, 0xbd2d7e2abec44c8eU,
  0x3d9d72e0bf6282d0U, 0x3e1277ce3e0b5138U, 0xbedfc1183d6df034U, 0xbdd08b6f3e7ceb97U,
  0x3ef3895e3e229577U, 0xbf98fcac3f6c9e97U, 0x3e2501f13eacf7efU, 0x3e12e2473f4c70f9U,
  0x3e01ecfdba198d1fU, 0x3ec940bfbe81488fU, 0x3e53436e3e8ad420U, 0x3eb7f877be512551U,
  0x3ea7a92dbf142c04U, 0x3db91e00be2d680eU, 0x3ced81a23f384020U, 0xbe9a9553bce6140eU,
  0xbf03125a3ea6046bU, 0x3ecec9ff3ece64a4U, 0x3cea291dbeccd05aU, 0x3f18b0df3e7e0e2fU,
  0xbedf6b38bed9359bU, 0x3e91e59cbca50440U, 0x3ef1de0e3ebe2df9U, 0x3f0883513e575744U,
  0x3e003fbd3e2406c6U, 0x3e6033793e7c69b5U, 0x3f0de0443ea7ad48U, 0xbdb2e1683e7263baU,
  0x3f0927a73e866af9U, 0x3ec1f6603e03d849U, 0x3f0e06e73f164229U, 0x3ebcb6a33f31adbcU,
  0x3ea248973e9114c3U, 0x3e82e1d83f01a93eU, 0x3f0395003f2a5355U, 0x3eb49d833dfd5f61U,
  0x3d2c83903f358e4bU, 0x3e41a09e3af5cc33U, 0x3f26e22b3c91bbf4U, 0x3e8f021a3e86cf44U,
  0x3ebdcf813b8c1464U, 0x3eb2d0e5bea60dafU, 0x3ebfe9803e4393ccU, 0x3e4835a4be251d49U,
  0xbd3fdf27bf095484U, 0xbd30ff683ee61e27U, 0x3e66fd503e8e865eU, 0x3ee10c053e8be2f2U,
  0xbf2810c3bdfa5f58U, 0x3ddc85003eb2c4fcU, 0x3e7d2ab0bf0e6407U, 0x3f2b49623efa1391U,
  0xbe22f1dcbf408aa4U, 0x3e9afb5a3e328601U, 0x3dcb407f3f1e706fU, 0xbde5293bbdf672bdU,
  0xbbf935dc3c553612U, 0xbd96544f3d4d231cU, 0x3f2ad7873f076c2eU, 0xbe3ef849beb68aa1U,
  0x3d570c2e3f1142f9U, 0xbf15f6f1bc41874eU, 0x3e205a2abee2289dU, 0xbe91e7393ddae886U,
  0x3e50feca3f07083fU, 0xbdde84773edad3acU, 0x3ead4002bf08a045U, 0xbf0359213d5170a9U,
  0x3b818ab2be79be22U, 0xbebffc313c99aa5cU, 0x3e1aed67bc855740U, 0xbbd8f8c2be4dc4f8U,
  0xbf38e0e53e9d7db9U, 0xbc5164ffbe117a26U, 0xbee7950d3d475546U, 0xbf3d2dcbbf5de0b7U,
  0x3e2af23c3e3c0ff5U, 0xbf71690c3c9c6e62U, 0x3df03345bf01307fU, 0xbdc422f6bc9e1309U,
  0xbe23b65abf6210c0U, 0x3e616c653da53395U, 0xbe4bdb513d09565cU, 0xbeaf39adbe089deaU,
  0x3d8f7fb33d1b4031U, 0xbf98963b3f9e2f56U, 0x3e07a5e4bee17b7bU, 0xbeda64a03df66361U,
  0xbd9ca9f9be875725U, 0xbd3b44fdbdc60a50U, 0xbea839033e574b63U, 0x3e80278d3f7948ebU,
  0xbe967a563f44f59aU, 0x3e776820bcc3bdbaU, 0x3f2b02eebe698b80U, 0x3e6e97efbeeb789cU,
  0x3f8bf32abde46ca0U, 0xbec7ff4bbc87c0e8U, 0x3eb429e83f8251d2U, 0xbeac0bfbbf858cd7U,
  0x3b8bca543eff90cbU, 0x3e9b22d8beec772fU, 0x3eb1b5b1bfa2c232U, 0xbe08221c3f0eb844U,
  0x3de297903eea30cbU, 0x3f5e66933ee94b24U, 0x3f8228193f690a5bU, 0x3f3196833f3fab56U,
  0x3f3ad9cc3ee742c6U, 0xbedf2679bea11878U, 0x3d2d364bbeb72e8bU, 0x3f28f3ed3f3f4684U,
  0xbed5febdbf713451U, 0xbf1804983f551f27U, 0x3f21d4e0bf84d1c0U, 0x3f3c52e23f5ca53fU,
  0x3e2d3fa63f1a24dcU, 0x3cbcc50fbda9b422U, 0x3eb144fdbdb200a9U, 0x3eb197df3f3529b2U,
  0xbe90a7be3e781e38U, 0x3df9627ebecdc7e7U, 0x3d8537ed3f5abc8aU, 0x3f0882b6be19aa62U,
  0x3ee49b86bef18ed4U, 0xbf30e0853d9bbb39U, 0xbfaa2f3c3e438210U, 0xbfa4b42b3f385110U,
  0x3ec172403e1b3ec1U, 0xbe43011b3e84f2fcU, 0xbf15494fbf07bfbcU, 0x3e2a7ce7be9631ecU,
  0xbf03f8c03ec512c5U, 0x3f50b1853f3ff4c5U, 0x3f104acd3f230505U, 0xbdcfaed3be2af12aU,
  0xbed4f26cbeeca276U, 0x3f8ebec0bf16fa00U, 0x3e0284a9bf07c060U, 0xbec79fa33e1ff011U,
  0xbf34bed7bf67b143U, 0xbc41a238bf3a2ff9U, 0x3e9a9e52bf14b4f4U, 0x3f53dfdfbf1b3d3fU,
  0xbe8c35993e39feafU, 0xbe881f6b3e743fb1U, 0xbf3df961bf05a77dU, 0x3e26267a3e0ea8d7U,
  0x3f3bc5a23e788ba7U, 0x3f6150cb3ec9dc77U, 0x3e84d8b23e981b8fU, 0x3c4df71ebcc1a151U,
  0x3f3681363e53853bU, 0x3f0ece35bfa71e98U, 0xbfc51e9dU,
};


ai_handle g_predictive_maintenance_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_predictive_maintenance_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

