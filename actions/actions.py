import time
import psycopg2
import chromadb
import google.generativeai as genai
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer

# Cấu hình cho LLM
GOOGLE_API_KEY = "AIzaSyC9kP8u9EcihOPBsyi7gd3lk5ahpO0DJlU"
genai.configure(api_key=GOOGLE_API_KEY)


llm_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash"  
)

# Model tạo vector
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class ActionAIConsultant(Action):
    def name(self) -> Text:
        return "action_ai_consultant"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # --- BẮT ĐẦU ĐO THỜI GIAN VÀ LOG ---
        start_time = time.time()
        print(f"\n{'='*30}")
        print(f"[START] Gọi hàm: {self.name()}")
        ##############

        user_message = tracker.latest_message.get('text')
        intent = tracker.latest_message.get('intent').get('name')
        print(f"User Message: '{user_message}' | Intent: '{intent}'")

        # Lấy các Slot để tăng độ chính xác cho AI
        loc = tracker.get_slot("destination")
        tm = tracker.get_slot("time")
        print(f"Slots nhận được -> Destination: {loc} | Time: {tm}")

        try:
            # 1. Truy vấn tri thức từ ChromaDB
            chroma_start = time.time() # Note thời gian

            client = chromadb.PersistentClient(path="/home/thinh/data_worker/chroma_db")
            collection = client.get_collection(name="vietnam_tourism")
            
            query_vec = embed_model.encode(user_message).tolist()
            results = collection.query(query_embeddings=[query_vec], n_results=3)

            context = ""
            if results and results['documents'] and results['documents'][0]:
                context = "\n".join(results['documents'][0])
            chroma_time = time.time() - chroma_start

            print(f"Truy vấn ChromaDB mất: {chroma_time:.2f}s")
            print(f"Dữ liệu context tìm được từ DB:\n{'-'*30}\n{context}\n{'-'*30}")

            # 2. Prompt Engineering dựa trên từng Intent cụ thể
            if intent == "ask_weather_timing":
                role = "Bạn là chuyên gia khí hậu du lịch."
            elif intent == "ask_location_feature":
                role = "Bạn là chuyên gia văn hóa và ẩm thực vùng miền."
            else:
                role = "Bạn là chuyên gia thiết kế tour du lịch."

            prompt = f"""
            {role}
            Dựa trên thông tin này: {context}
            Câu hỏi: "{user_message}"
            Địa điểm khách quan tâm: {loc if loc else 'Chưa rõ'}
            Thời gian khách muốn đi: {tm if tm else 'Chưa rõ'}
            
            Hãy trả lời ngắn gọn, thân thiện và tập trung vào thông tin {loc if loc else 'được hỏi'}. 
            Nếu có chi phí (Avg_Cost), hãy nhắc đến để khách tham khảo.
            """

            # 3. Gọi Gemini sinh nội dung
            response = llm_model.generate_content(prompt)
            dispatcher.utter_message(text=response.text)

        except Exception as e:
            print(f"LỖI TRONG ACTION: {str(e)}")
            dispatcher.utter_message(text=f"Lỗi hệ thống tư vấn: {str(e)}")
        
        # --- KẾT THÚC ĐO THỜI GIAN ---
        total_time = time.time() - start_time
        print(f"⏱️ [END] Tổng thời gian phản hồi: {total_time:.2f}s")
        print(f"{'='*30}\n")

        return []

class ActionSearchTourInfo(Action):
    def name(self) -> Text:
        return "action_search_tour_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # --- BẮT ĐẦU ĐO THỜI GIAN VÀ LOG ---
        start_time = time.time()
        print(f"\n{'='*50}")
        print(f"[START] Gọi hàm: {self.name()}")

        dest = tracker.get_slot("destination")
        cat = tracker.get_slot("category")  # Bắt thêm category
        
        # In ra màn hình terminal 2 để dễ debug
        print(f">>> DEBUG - Destination: {dest} | Category: {cat}")

        if not dest:
            dispatcher.utter_message(text="Bạn muốn tìm thông tin dịch vụ tại điểm đến nào nhỉ?")
            print(f"Thiếu Slot Destination, hủy truy vấn SQL.")
            print(f"[END] Tổng thời gian phản hồi: {time.time() - start_time:.2f}s")
            print(f"{'='*50}\n")
            return []

        try:
            sql_start = time.time() # Đo thời gian

            conn = psycopg2.connect(database="tourism_db", user="thinh", password="123456", host="127.0.0.1", port="5432")
            cur = conn.cursor()
            
            # Khởi tạo câu query cơ bản
            query = "SELECT name, address, description FROM destinations WHERE city ILIKE %s"
            params = [f"%{dest}%"]
            
            # Nếu người dùng có nhắc đến loại hình (khách sạn, bãi biển...), thêm điều kiện lọc
            if cat:
                query += " AND category = %s"
                params.append(cat)
            
            print(f"🗄️ Đang chạy SQL Query: {query} với Params: {params}")
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            sql_time = time.time() - sql_start
            print(f"Truy vấn PostgreSQL mất: {sql_time:.2f}s | Tìm thấy {len(rows)} kết quả.")

            if rows:
                # Dịch ngược category sang tiếng Việt
                cat_vn = "dịch vụ"
                if cat == 'hotel': cat_vn = "khách sạn/resort"
                elif cat == 'beach': cat_vn = "bãi biển"
                elif cat == 'restaurant': cat_vn = "nhà hàng/quán ăn"
                elif cat == 'pagoda': cat_vn = "địa điểm tâm linh"

                msg = f"Đây là các {cat_vn} tại {dest} mình tìm được:\n"
                for r in rows:
                    msg += f"- **{r[0]}**: {r[1]} ({r[2]})\n"
                dispatcher.utter_message(text=msg)
            else:
                dispatcher.utter_message(text=f"Hiện tại mình chưa tìm thấy thông tin phù hợp tại {dest}.")
            
            cur.close()
            conn.close()
        except Exception as e:
            print(f"LỖI SQL: {str(e)}")
            dispatcher.utter_message(text=f"Lỗi tra cứu database: {e}")

        # --- KẾT THÚC ĐO THỜI GIAN ---
        total_time = time.time() - start_time
        print(f"[END] Tổng thời gian phản hồi: {total_time:.2f}s")
        print(f"{'='*30}\n")

        return []