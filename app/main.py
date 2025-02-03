# Import required libraries
from dash import html
from dash import dcc
from dash import  html, Input, Output, State
import dash_bootstrap_components as dbc
from dash import Dash
from dash_chat import ChatComponent
import torch
import model

text_transform,token_transform,vocab_transform, mapping = model.get_trasforms()
device = model.device
SRC_LANGUAGE =  model.SRC_LANGUAGE
TRG_LANGUAGE =  model.TRG_LANGUAGE
SOS_IDX = model.SOS_IDX
EOS_IDX = model.EOS_IDX
special_symbols = model.special_symbols
model = model.create_model()
model.load_state_dict(torch.load('./model/gen.pt',map_location=torch.device('cpu')))

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


# Create a dash application
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    dcc.Tabs(
        [
            dcc.Tab(
                label="A3 - Machine Translation",
                children=[
                    html.H2(
                        children="English - Myanmar",
                        style={
                            "textAlign": "center",
                            "color": "#503D36",
                            "font-size": 40,
                        },
                    ),
                    dbc.Stack(
                        dcc.Textarea(
                            id="note",
                            value="""Note : This model can be used to translate from English to Myanmar""",
                            style={
                                "width": "100%",
                                "height": 15,
                                "whiteSpace": "pre-line",
                                "textAlign": "center",
                            },
                            readOnly=False,
                        )
                    ),                    
                    html.Br(),
                   ChatComponent(
        id="chat-component",
        messages=[
            {"role": "assistant", "content": "Hello!, I can translate from English to Myanmar."},
        ],
    ),
                ],
            ),
        ]
    )
)



def infer(src, text_transform,vocab_transform,model,max_len=100):
    src_text = text_transform[SRC_LANGUAGE](src).to(device).unsqueeze(0) 
    # trg_text = text_transform[TRG_LANGUAGE](src).to(device)

    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)

    model.eval()

    trg_idx = [SOS_IDX]


    for i in range(max_len):
        with torch.no_grad():
            output, attentions = model(src_text, torch.LongTensor(trg_idx).unsqueeze(0).to(device)) #turn off teacher forcing

        pred_token = output.argmax(2)[:, -1].item()
        trg_idx.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
    
    # print(trg_indexes)
    trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[i] for i in trg_idx]
    return trg_tokens[1:], attentions  # E


@app.callback(
    Output("chat-component", "messages"),
    Input("chat-component", "new_message"),
    State("chat-component", "messages"),
    prevent_initial_call=True,
)
def handle_chat(new_message, messages):
    if not new_message:
        return messages

    updated_messages = messages + [new_message]
    
    if new_message["role"] == "user":
        translation,_ = infer(new_message["content"], text_transform,vocab_transform,model,100)
        for rm in special_symbols:
            while rm in translation:
                translation.remove(rm)
        bot_response = {"role": "assistant", "content": ' '.join(translation)}
        # bot_response = {"role": "assistant", "content": response.choices[0].message.content.strip()}
        return updated_messages + [bot_response]

    
    # updated_messages += new_message + " testing"

    return updated_messages

# Run the app
if __name__ == "__main__":
    app.run_server()
