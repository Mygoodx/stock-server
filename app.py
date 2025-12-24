import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import re
import os
from datetime import timedelta

app = Flask(__name__)
CORS(app)

# 1. 자주 찾는 미국 주식 하드코딩
STOCK_MAP = {
    # [M7 및 빅테크]
    "애플": "AAPL", "마소": "MSFT", "마이크로소프트": "MSFT",
    "엔비디아": "NVDA", "구글": "GOOGL", "알파벳": "GOOGL",
    "아마존": "AMZN", "테슬라": "TSLA", "메타": "META", "페이스북": "META",
    "넷플릭스": "NFLX", "오라클": "ORCL", "어도비": "ADBE",
    "세일즈포스": "CRM", "아이비엠": "IBM",
    
    # [반도체]
    "에이엠디": "AMD", "AMD": "AMD", "인텔": "INTC", 
    "TSMC": "TSM", "티에스엠씨": "TSM",
    "브로드컴": "AVGO", "퀄컴": "QCOM", "마이크론": "MU",
    "텍사스인스트루먼트": "TXN", "어플라이드머티리얼즈": "AMAT",
    "램리서치": "LRCX", "ASML": "ASML", "아스몰": "ASML",
    "ARM": "ARM", "암": "ARM", "슈퍼마이크로": "SMCI",
    
    # [성장주/AI/소프트웨어]
    "아이온큐": "IONQ", "팔란티어": "PLTR", "유니티": "U",
    "로블록스": "RBLX", "코인베이스": "COIN", "에어비앤비": "ABNB",
    "우버": "UBER", "리비안": "RIVN", "루시드": "LCID",
    "스노우플레이크": "SNOW", "크라우드스트라이크": "CRWD",
    "팔로알토": "PANW", "데이터독": "DDOG", "코어위브": "COREWEAVE_IPO_PENDING",
    
    # [소비재/브랜드]
    "코카콜라": "KO", "펩시": "PEP", "스타벅스": "SBUX",
    "나이키": "NKE", "디즈니": "DIS", "맥도날드": "MCD",
    "월마트": "WMT", "코스트코": "COST", "홈디포": "HD",
    "프록터앤갬블": "PG", "존슨앤존슨": "JNJ",
    
    # [헬스케어/비만치료제]
    "일라이릴리": "LLY", "노보노디스크": "NVO",
    "화이자": "PFE", "머크": "MRK", "암젠": "AMGN",
    
    # [금융/결제]
    "제이피모건": "JPM", "뱅크오브아메리카": "BAC",
    "비자": "V", "마스터카드": "MA", "페이팔": "PYPL",
    "블록": "SQ", "버크셔해서웨이": "BRK.B",
    
    # [방산/우주]
    "록히드마틴": "LMT", "보잉": "BA", "알티엑스": "RTX",
    "제너럴일렉트릭": "GE",
    
    # [한국 관련]
    "쿠팡": "CPNG",
    
    # [ETF - 지수/배당]
    "스파이": "SPY", "SPY": "SPY", "VOO": "VOO", "IVV": "IVV",
    "큐큐큐": "QQQ", "QQQ": "QQQ",
    "슈드": "SCHD", "SCHD": "SCHD", "제피": "JEPI",
    "배당귀족": "NOBL", "리얼티인컴": "O",
    
    # [ETF - 레버리지/반도체]
    "티큐": "TQQQ", "TQQQ": "TQQQ", 
    "속슬": "SOXL", "SOXL": "SOXL",
    "삭스": "SOXX", "반도체ETF": "SOXX"
}

def is_valid_ticker(symbol):
    if not symbol: return False
    return bool(re.match(r'^[A-Z\.-]+$', symbol))

def search_ticker(query):
    query = query.strip()
    if query in STOCK_MAP: 
        if STOCK_MAP[query] == "COREWEAVE_IPO_PENDING":
            return None, "코어위브 (상장 예정/비상장)"
        return STOCK_MAP[query], query
        
    if query.upper().isalpha() and 2 <= len(query) <= 5: return query.upper(), query.upper()

    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {'q': query, 'quotesCount': 1, 'newsCount': 0, 'region': 'US'}
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, params=params, headers=headers, timeout=3)
        data = res.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            symbol = data['quotes'][0]['symbol']
            name = data['quotes'][0].get('shortname', symbol)
            if not symbol.endswith('.KS') and not symbol.endswith('.KQ'):
                if is_valid_ticker(symbol): return symbol, name
    except: pass

    try:
        ac_url = "https://ac.finance.naver.com/ac"
        params = {'q': query, 'q_enc': 'euc-kr', 'st': '111', 'r_format': 'json', 'r_enc': 'euc-kr'}
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(ac_url, params=params, headers=headers, timeout=3)
        data = res.json()
        if 'items' in data and len(data['items']) > 0 and len(data['items'][0]) > 0:
            for item in data['items'][0]:
                code = item[0]
                name = item[1]
                if re.search(r'[A-Z]+\.[ONAK]', code):
                    ticker = code.split('.')[0]
                    if is_valid_ticker(ticker): return ticker, name
    except: pass
    return None, None

def get_yahoo_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # 1. 재무 데이터
        financials = stock.quarterly_financials
        if financials.empty:
            print(f"No financial data for {ticker_symbol}")
            return None
        
        # 2. 과거 주가 데이터
        hist_price = stock.history(period="2y")
        
        df = financials.T
        df = df.sort_index(ascending=True).iloc[-5:]
        
        needed_rows = {
            'revenue': ['Total Revenue', 'Operating Revenue', 'Revenue'],
            'op_income': ['Operating Income', 'Operating Profit'],
            'eps': ['Basic EPS', 'Diluted EPS', 'Basic Earnings Per Share']
        }
        
        try:
            info = stock.info
            shares_outstanding = info.get('sharesOutstanding')
            current_per = info.get('trailingPE') or info.get('forwardPE') or 20.0
            current_psr = info.get('priceToSalesTrailing12Months') or 5.0
            # [추가] 현재 주가 가져오기
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        except:
            shares_outstanding = None
            current_per = 20.0
            current_psr = 5.0
            current_price = None

        parsed_data = []

        for date, row in df.iterrows():
            period_str = date.strftime('%Y-%m')
            
            def find_val(keys, row_data):
                for k in keys:
                    if k in row_data: return row_data[k]
                return None

            rev = find_val(needed_rows['revenue'], row)
            op = find_val(needed_rows['op_income'], row)
            eps = find_val(needed_rows['eps'], row)
            
            rev = float(rev) if rev is not None else None
            op = float(op) if op is not None else None
            eps = float(eps) if eps is not None else None
            
            # 과거 실제 주가 및 PER 찾기
            actual_price = None
            historical_per = None
            
            if not hist_price.empty:
                try:
                    target_dates = [date + timedelta(days=i) for i in range(-3, 3)]
                    for d in target_dates:
                        d_str = d.strftime('%Y-%m-%d')
                        if d_str in hist_price.index:
                            actual_price = hist_price.loc[d_str]['Close']
                            break
                    if actual_price is None:
                        idx = hist_price.index.get_indexer([date], method='nearest')[0]
                        actual_price = hist_price.iloc[idx]['Close']
                except:
                    actual_price = None

            if actual_price and eps and eps > 0:
                historical_per = actual_price / (eps * 4)
            elif actual_price and rev and shares_outstanding:
                sps = rev / shares_outstanding
                historical_per = actual_price / (sps * 4) # PSR
            
            applied_multiple = historical_per if historical_per else (current_per if (eps and eps > 0) else current_psr)

            parsed_data.append({
                'period': period_str,
                'revenue': rev,
                'op_income': op,
                'eps': eps,
                'per': applied_multiple, 
                'calc_price': actual_price, 
                'is_actual': True, 
                'method': "Actual" if historical_per else "Est."
            })
            
        if not parsed_data: return None

        # 미래 예측 로직
        pdf = pd.DataFrame(parsed_data)
        pdf = pdf.ffill().bfill()
        
        numeric_cols = ['revenue', 'eps', 'op_income']
        for col in numeric_cols:
            if col in pdf.columns:
                pdf[col] = pd.to_numeric(pdf[col], errors='coerce')
        
        # 성장률 계산
        growth_rates = pdf[numeric_cols].pct_change(fill_method=None).mean().fillna(0)
        last = pdf.iloc[-1]
        
        pred_revenue = last['revenue'] * (1 + growth_rates['revenue']) if pd.notna(last['revenue']) else None
        pred_eps = last['eps'] * (1 + growth_rates['eps']) if pd.notna(last['eps']) else None
        # [추가] 영업이익 예측 계산
        pred_op_income = last['op_income'] * (1 + growth_rates['op_income']) if pd.notna(last['op_income']) else None
        
        # [수정] 직전 분기의 실제 PER/PSR을 초기 예측 멀티플로 사용
        last_hist = parsed_data[-1]
        pred_per = last_hist['per']
        
        pred_price = None
        method = "PER"
        
        if pred_eps is not None:
            if pred_eps > 0:
                # 흑자일 때: PER 방식
                pred_price = pred_eps * 4 * pred_per
                method = "PER"
            elif pred_revenue is not None and shares_outstanding:
                 # 적자일 때: PSR 방식 (직전 분기도 적자였다면 pred_per는 PSR값일 것임)
                 sps = pred_revenue / shares_outstanding
                 pred_price = sps * 4 * pred_per
                 method = "PSR"
        
        prediction = {
            'period': 'Next Q (Pred)',
            'revenue': round(pred_revenue) if pred_revenue else None,
            'op_income': round(pred_op_income) if pred_op_income else None, # [추가] 영업이익
            'eps': round(pred_eps, 2) if pred_eps else None,
            'per': round(pred_per, 2),
            'calc_price': round(pred_price, 2) if pred_price else None, # 계산된 예측 주가 그대로 사용
            'is_actual': False,
            'method': method
        }
        
        return {
            'history': parsed_data,
            'prediction': prediction,
            'info': {
                'shares': shares_outstanding, 
                'currency': 'USD',
                'current_price': current_price # [추가] 현재 주가
            }
        }

    except Exception as e:
        print(f"Yahoo Data Error: {e}")
        return None

def convert_nan_to_none(obj):
    if isinstance(obj, dict): return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
    elif pd.isna(obj): return None
    else: return obj

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    stock_name = data.get('stock_name')
    if not stock_name: return jsonify({'error': '종목명을 입력해주세요.'}), 400
        
    print(f"Searching: {stock_name}")
    ticker, real_name = search_ticker(stock_name)
    
    if not ticker:
        if "비상장" in real_name:
            return jsonify({'error': f"'{stock_name}'은(는) 상장 예정이거나 비상장 기업입니다."}), 404
        return jsonify({'error': f"'{stock_name}' 정보 없음"}), 404
    
    if not is_valid_ticker(ticker):
         return jsonify({'error': f"유효하지 않은 티커: {ticker}"}), 404
        
    print(f"Ticker: {ticker}")
    result = get_yahoo_data(ticker)
    if not result:
        return jsonify({'error': f"'{real_name}' 데이터 없음"}), 404
        
    return jsonify({
        'stock_name': real_name,
        'code': ticker,
        'data': convert_nan_to_none(result)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"=== US Stock Server Started on Port {port} ===")
    app.run(host='0.0.0.0', port=port, debug=True)
