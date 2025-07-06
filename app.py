from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from datetime import datetime
import time

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])

# SecondLife特有の設定
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Hugging Face無料APIの設定
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
HF_HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', 'your_huggingface_token')}"}

# 日本語対応のモデル（無料）
JAPANESE_MODEL_URL = "https://api-inference.huggingface.co/models/rinna/japanese-gpt-neox-3.6b-instruction-sft"

# 会話履歴を保存（簡易版）
conversation_history = {}

@app.route('/')
def home():
    return "SecondLife AI Chatbot Server is running!"

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    # OPTIONSリクエストへの対応
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        # リクエストデータの検証
        if not request.is_json:
            return jsonify({"response": "JSONフォーマットが必要です"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"response": "データが空です"}), 400
            
        message = data.get('message', '').strip()
        user = data.get('user', 'anonymous')
        
        if not message:
            return jsonify({"response": "メッセージが空です"}), 400
        
        print(f"[{datetime.now()}] User {user}: {message}")
        
        # 処理時間を記録
        start_time = time.time()
        
        # 日本語でのAI応答生成
        response = generate_japanese_response(message, user)
        
        processing_time = time.time() - start_time
        print(f"[{datetime.now()}] Processing time: {processing_time:.2f}s")
        
        # 会話履歴に追加
        if user not in conversation_history:
            conversation_history[user] = []
        conversation_history[user].append({"user": message, "bot": response})
        
        # 履歴を最新10件に制限
        if len(conversation_history[user]) > 10:
            conversation_history[user] = conversation_history[user][-10:]
        
        print(f"[{datetime.now()}] Bot response: {response}")
        
        # レスポンスヘッダーを設定
        response_obj = jsonify({
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "processing_time": processing_time
        })
        
        response_obj.headers.add('Access-Control-Allow-Origin', '*')
        response_obj.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        response_obj.headers.add('Pragma', 'no-cache')
        response_obj.headers.add('Expires', '0')
        
        return response_obj
    
    except Exception as e:
        print(f"Error: {str(e)}")
        error_response = jsonify({
            "response": "申し訳ございません。エラーが発生しました。",
            "error": str(e)
        })
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

def generate_japanese_response(message, user):
    """日本語でのAI応答生成"""
    try:
        # シンプルなルールベース応答（フォールバック）
        simple_responses = {
            "hello": "こんにちは！お元気ですか？",
            "hi": "こんにちは！何かお手伝いできることはありますか？",
            "こんにちは": "こんにちは！素敵な日ですね。",
            "おはよう": "おはようございます！今日も良い一日になりそうですね。",
            "こんばんは": "こんばんは！お疲れ様でした。",
            "元気": "私は元気です！あなたはいかがですか？",
            "ありがとう": "どういたしまして！お役に立てて嬉しいです。",
            "さようなら": "さようなら！また話しましょう。",
            "名前": "私は AI チャットボットです。よろしくお願いします！",
            "天気": "天気について具体的な情報は持っていませんが、きっと素敵な天気でしょう！",
            "時間": f"現在時刻は {datetime.now().strftime('%H:%M')} です。",
            "日付": f"今日は {datetime.now().strftime('%Y年%m月%d日')} です。"
        }
        
        message_lower = message.lower()
        
        # キーワードマッチング
        for keyword, response in simple_responses.items():
            if keyword in message_lower:
                return response
        
        # Hugging Face APIを試す
        try:
            response = query_huggingface(message)
            if response:
                return response
        except Exception as e:
            print(f"Hugging Face API error: {e}")
        
        # デフォルト応答
        default_responses = [
            "とても興味深いお話ですね。もう少し詳しく教えていただけますか？",
            "なるほど、そうですね。他にも何かお聞きしたいことはありますか？",
            "それについて考えてみますね。どう思われますか？",
            "面白い視点ですね。もう少し詳しく聞かせてください。",
            "そうですね。私も同じように思います。",
            "それは素晴らしいアイデアだと思います！",
            "とても参考になりました。ありがとうございます。"
        ]
        
        import random
        return random.choice(default_responses)
        
    except Exception as e:
        print(f"Response generation error: {e}")
        return "申し訳ございません。うまく理解できませんでした。もう一度お話しいただけますか？"

def query_huggingface(message):
    """Hugging Face APIクエリ"""
    try:
        # 日本語用のプロンプト
        prompt = f"ユーザー: {message}\nAI: "
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 100,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9
            }
        }
        
        # まず日本語モデルを試す
        response = requests.post(JAPANESE_MODEL_URL, headers=HF_HEADERS, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # プロンプト部分を除去
                ai_response = generated_text.replace(prompt, '').strip()
                if ai_response:
                    return ai_response
        
        # 英語モデルでも試す
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                ai_response = generated_text.replace(prompt, '').strip()
                if ai_response:
                    # 簡単な英日翻訳（基本的な挨拶のみ）
                    translations = {
                        "hello": "こんにちは",
                        "hi": "こんにちは",
                        "good": "良い",
                        "thank you": "ありがとう",
                        "yes": "はい",
                        "no": "いいえ"
                    }
                    
                    for en, jp in translations.items():
                        ai_response = ai_response.replace(en, jp)
                    
                    return ai_response
        
        return None
        
    except Exception as e:
        print(f"Hugging Face query error: {e}")
        return None

@app.route('/history/<user>')
def get_history(user):
    """会話履歴を取得"""
    return jsonify(conversation_history.get(user, []))

@app.route('/clear_history/<user>', methods=['POST'])
def clear_history(user):
    """会話履歴をクリア"""
    if user in conversation_history:
        del conversation_history[user]
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
