# Contents of ~/my_app/streamlit_app.py
import streamlit as st

from typing import Dict, List

from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer,
    BertTokenizer,
    PreTrainedTokenizerFast,
)

from lib.tokenization_kobert import KoBertTokenizer


class KoGPT2Tokenizer(PreTrainedTokenizerFast):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]


if "tokenizer" not in st.session_state:
    src_tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    trg_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    # src_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # trg_tokenizer = KoGPT2Tokenizer.from_pretrained(
    #     "skt/kogpt2-base-v2",
    #     bos_token="</s>",
    #     eos_token="</s>",
    #     unk_token="<unk>",
    #     pad_token="<pad>",
    #     mask_token="<mask>",
    # )
    st.session_state.tokenizer = src_tokenizer, trg_tokenizer
else:
    src_tokenizer, trg_tokenizer = st.session_state.tokenizer


@st.cache
def get_model(bos_token_id):
    model = EncoderDecoderModel.from_pretrained("dump/baseline_best_model")
    # model = EncoderDecoderModel.from_pretrained("dump/first_model")
    model.config.decoder_start_token_id = bos_token_id
    model.eval()
    # model.cuda()

    return model


model = get_model(trg_tokenizer.bos_token_id)

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈\nTEAM_5의 영한번역기 googoorm입니다")

st.title("googoorm")
st.subheader("영-한 번역기에 오신 것을 환영합니다!\n자연스러운 구어체 번역을 도와드립니다")

kor = st.text_area("입력", placeholder="번역할 영어")

if st.button("번역!", help="해당 영어 입력을 번역합니다."):
    embeddings = src_tokenizer(
        kor,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    embeddings = {k: v for k, v in embeddings.items()}
    # embeddings = {k: v.cuda() for k, v in embeddings.items()}
    output = model.generate(**embeddings)[0, 1:-1].cpu()
    st.text_area("출력", value=trg_tokenizer.decode(output), disabled=True)
    # st.text_area("출력", value="안녕하세요", disabled=True)
