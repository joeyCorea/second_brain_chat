from second_brain_chat.chat_bot import chain

def test_basic_greeting():
    res = chain.invoke("Hi")
    assert res["response"].strip() != ""

