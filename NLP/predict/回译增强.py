from google_trans_new import google_translator
# 实例化翻译对象
translator = google_translator()
# 进⾏第⼀次批量翻译, 翻译⽬标是韩语
text = ["这家价格很便宜", "这家价格很便宜"]
ko_res = translator.translate(text, lang_src="zh-cn", lang_tgt="ko")
# 打印结果
print("中间翻译结果:")
print(ko_res)
## 最后在翻译回中⽂, 完成回译全部流程
cn_res = translator.translate(ko_res, lang_src='ko', lang_tgt='zh-cn')
print("回译得到的增强数据:")
print(cn_res)