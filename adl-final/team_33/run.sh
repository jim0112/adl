pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
wget -O best_model.ckpt "https://www.dropbox.com/s/o127npoxeh56g1u/RL_model_longperson_1000.ckpt?dl=1"
python3 simulator.py --split test --disable_output_dialog --num_chats 980 --disable_output_dialog
