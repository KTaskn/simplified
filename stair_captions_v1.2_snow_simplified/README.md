# やさしい日本語版 STAIR Captions v1.2 
[やさしい日本語2,000語](https://www.jnlp.org/GengoHouse/list/%E8%AA%9E%E5%BD%99)の語彙をベースに[STAIR Captions v1.2](http://captions.stair.center/)のキャプションを平易化したコーパスです

## 平易化には下記を実施しました
- [独自の翻訳モデル](https://huggingface.co/KTaskn/t5-base-japanese-snow-extended)を利用した平易化
- 語彙を機械的に置換
- 人力作業による平易化
- LLMを用いた平易化

## 語彙の検証
付属のcheck_tokens.pyの検証を行なっています．
一部STAIRに頻出する語彙で，**やさしい日本語2,000語**に含まれない語彙を追加しています．
また数値に関しては対象外としています．

## 平易化結果の検証
細かな検証は行えていません