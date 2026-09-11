# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0, các lần chạy cho ra câu trả lời gần như y hệt nhau về nội dung lẫn cách diễn đạt — model chọn phương án "an toàn" nhất mỗi lần. Càng tăng temperature lên 0.5, 1.0 rồi 1.5, cách dùng từ, ví dụ đưa ra và cấu trúc câu càng trở nên phong phú, ít lặp lại hơn giữa các lần gọi. Đổi lại, ở mức 1.5 câu trả lời bắt đầu có dấu hiệu lan man hoặc thiếu nhất quán, cho thấy độ sáng tạo tăng lên đi kèm rủi ro giảm độ ổn định.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi nghiêng về khoảng mức 0.2–0.4. Ở khoảng này, câu trả lời vẫn giữ được độ ổn định và bám sát thông tin chính xác — điều quan trọng nhất với một chatbot chăm sóc khách hàng — nhưng vẫn đủ linh hoạt để không nghe cứng nhắc hay máy móc khi giao tiếp.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> So sánh giá output: GPT-4o là 0,010 USD/1K token, GPT-4o-mini là 0,0006 USD/1K token — chênh lệch khoảng 16,7 lần. Nhân với quy mô 10.000 người dùng × 3 lượt/ngày × 350 token, tổng cộng workload này tạo ra khoảng 10,5 triệu token đầu ra mỗi ngày. GPT-4o đáng chọn cho các tác vụ đòi hỏi suy luận sâu hoặc độ chính xác cao; GPT-4o-mini hợp lý hơn cho các tác vụ lặp đi lặp lại như phân loại yêu cầu hay trả lời câu hỏi thường gặp, nơi việc tiết kiệm chi phí quan trọng hơn phần chênh lệch chất lượng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phiên bản giáo viên tiểu học cho ra câu trả lời ngắn gọn, dùng từ ngữ đơn giản, kèm những phép so sánh gần gũi dễ hình dung với trẻ nhỏ. Phiên bản chuyên gia tài chính thì dài hơn, sử dụng thuật ngữ chuyên ngành và đi sâu vào chi tiết kỹ thuật hơn hẳn. Qua đó có thể thấy system prompt là yếu tố quyết định vai trò, đối tượng người nghe, giọng điệu và độ chuyên sâu mà model thể hiện trong câu trả lời — dù câu hỏi gốc không đổi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn 106 từ, công thức ước lượng 106/0.75 cho ra khoảng 141 token, trong khi tiktoken (GPT-4o) đếm thực tế chỉ 134 token — chênh khoảng 5,5%, tức công thức ước lượng hơi cao so với thực tế trong trường hợp này. Dù vậy, so với bản dịch tiếng Anh cùng nội dung (chỉ 109 token), tiếng Việt vẫn tốn nhiều token hơn rõ rệt — nguyên nhân là bộ mã hóa BPE được huấn luyện chủ yếu trên dữ liệu tiếng Anh, nên các từ tiếng Việt có dấu thường bị tách vụn thành nhiều token con thay vì gọn trong một token như từ tiếng Anh phổ biến.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming phát huy giá trị rõ nhất ở những nơi người dùng phải chờ phản hồi dài và trực tiếp tương tác — như chatbot, trợ lý học tập, hay công cụ hỗ trợ viết — vì nó giúp giảm cảm giác chờ đợi bằng cách hiển thị nội dung ngay khi có. Ngược lại, non-streaming hợp lý hơn khi ứng dụng cần toàn bộ kết quả hoàn chỉnh trước khi xử lý tiếp — chẳng hạn lưu response vào database, chạy tác vụ nền không ai theo dõi trực tiếp, hoặc khi câu trả lời quá ngắn nên streaming không tạo ra khác biệt đáng kể.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff kéo giãn khoảng cách giữa các lần retry sau mỗi lần thất bại, nhờ vậy giảm bớt áp lực dồn lên API đang quá tải và cho hệ thống có thời gian hồi phục. Nếu tất cả client đều dùng delay cố định giống nhau, chúng sẽ đồng loạt gửi lại request đúng vào cùng một thời điểm, tạo thành từng đợt request dồn dập — hiện tượng thường gọi là "thundering herd" — khiến tình trạng quá tải tiếp tục lặp lại thay vì giảm đi. Thêm jitter ngẫu nhiên vào delay sẽ giúp các client không retry trùng thời điểm với nhau, càng giảm rủi ro này.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona được chọn là một trợ giảng lập trình thân thiện, với system prompt: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." Cụm từ "bằng tiếng Việt" được thêm vào có chủ đích để đảm bảo đầu ra luôn khớp với ngôn ngữ mà người học trong khóa sử dụng, tránh trường hợp model tự chuyển sang tiếng Anh khi gặp câu hỏi mơ hồ.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất nằm ở việc history chỉ giữ được ba lượt hội thoại gần nhất, khiến trợ lý dễ đánh mất mục tiêu hoặc các quyết định đã thống nhất từ những lượt trước đó. Hướng cải thiện là bổ sung một lớp bộ nhớ tóm tắt: cứ sau vài lượt trò chuyện, gọi model để tạo một đoạn tóm tắt ngắn về mục tiêu, sở thích và các quyết định quan trọng của người dùng, rồi đưa đoạn tóm tắt này vào cùng system context với phần history hiện có ở mỗi lượt gọi tiếp theo.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
