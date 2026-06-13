from locust import HttpUser, task, between

class TravelWebsiteUser(HttpUser):
    # Thời gian chờ ngẫu nhiên giữa các request của một user ảo (1 - 3 giây)
    wait_time = between(1.0, 3.0)

    @task(62)
    def view_tours(self):
        """Mô phỏng hành vi xem danh sách Tour (Tỷ trọng: 62%)"""
        self.client.get("/tours", name="GET /tours")

    @task(25)
    def view_blog(self):
        """Mô phỏng hành vi đọc bài viết Blog (Tỷ trọng: 35%)"""
        self.client.get("/blog/posts", name="GET /blog/posts")

    @task(3)
    def chat_with_bot(self):
        """Mô phỏng hành vi chat với Chatbot (Tỷ trọng: 3%)"""
        payload = {
            "sender": "locust_user",
            "message": "Tôi muốn đi du lịch Đà Lạt"
        }
        self.client.post("/chat/rasa", json=payload, name="POST /chat/rasa")

    @task(10)
    def health_check(self):
        """Mô phỏng request nhẹ/Health check qua API Gateway (Tỷ trọng: 10%)"""
        self.client.get("/", name="GET / (Homepage/Health)")
