"""Turns an intent into a reply: template text, plus job hits from search-svc for job requests.

Template answers only. Nothing here calls a language model or a hosted API.
"""
import logging
from dataclasses import dataclass, field

from app.chat import intent as I
from app.chat.search import SearchClient
from app.errors import SearchUnavailable

log = logging.getLogger("ai.chat")

DISCLAIMER = "Nội dung do hệ thống tạo theo mẫu, hãy kiểm tra lại tin tuyển dụng trước khi ứng tuyển."

TEXT: dict[str, str] = {
    I.GREETING: "Chào bạn! Mình là trợ lý việc làm của Job Platform. Bạn muốn tìm việc theo kỹ năng, vị trí hay địa điểm nào?",
    I.HELP: (
        "Mình có thể giúp bạn:\n"
        "- Tìm việc theo kỹ năng, vị trí và địa điểm, ví dụ: \"tìm việc Java ở Hà Nội\"\n"
        "- Gợi ý cách viết CV\n"
        "- Mẹo chuẩn bị phỏng vấn\n"
        "Mình trả lời theo quy tắc và mẫu có sẵn, chưa hiểu ngữ cảnh của các câu hỏi trước đó."
    ),
    I.OFF_TOPIC: "Mình chỉ hỗ trợ câu hỏi về việc làm, CV và nghề nghiệp. Hãy hỏi về vị trí, kỹ năng hoặc mẹo phỏng vấn nhé.",
    I.CV: (
        "CV nên nêu rõ kỹ năng đúng với mô tả công việc, dùng số liệu cụ thể (ví dụ: giảm 30% thời gian xử lý), "
        "và có 3-5 dòng tóm tắt mục tiêu. Với mỗi vị trí, liệt kê stack, số năm kinh nghiệm và thành tích đo được."
    ),
    I.INTERVIEW: (
        "Chuẩn bị 3 dự án gần nhất theo khung STAR (bối cảnh, nhiệm vụ, hành động, kết quả). "
        "Đọc kỹ mô tả công việc để lấy từ khóa, và chuẩn bị câu hỏi về team, quy trình và kỳ vọng 3 tháng đầu. "
        "Lương nên bàn sau khi hai bên đã hiểu rõ phạm vi công việc."
    ),
    I.UNKNOWN: (
        "Mình chưa hiểu câu hỏi này. Bạn có thể hỏi theo vị trí, kỹ năng hoặc địa điểm, "
        "ví dụ: \"tìm việc React ở Đà Nẵng\", hoặc hỏi về CV và phỏng vấn."
    ),
}
NO_CRITERIA = "Bạn muốn tìm việc gì? Hãy cho mình vị trí, kỹ năng hoặc địa điểm, ví dụ: \"tìm việc kế toán ở Hà Nội\"."
SEARCH_FAILED = "Hiện mình chưa tra cứu được tin tuyển dụng. Bạn vui lòng thử lại sau ít phút."
SEARCH_FAILED_CODE = "search_service_unavailable"


@dataclass
class Reply:
    text: str
    jobs: list[dict] = field(default_factory=list)  # ChatJobHint: id, title, company
    error: dict | None = None  # {code, message}: sent as an SSE error event


def describe(found: I.Intent) -> str:
    """'Java, React ở Hà Nội' style label for the criteria the user asked for."""
    head = " ".join(part for part in [found.keyword, ", ".join(found.skills)] if part)
    return f"{head} tại {found.location}" if found.location and head else (head or f"tại {found.location}")


def _job_text(found: I.Intent, jobs: list[dict]) -> str:
    if not jobs:
        return f"Hiện chưa có tin nào khớp với {describe(found)}. Bạn thử bỏ bớt một tiêu chí hoặc dùng từ khóa khác nhé.\n{DISCLAIMER}"
    lines = [f"Mình tìm thấy {len(jobs)} tin phù hợp với {describe(found)}:"]
    lines += [f"- {job['title']} tại {job['company']}" for job in jobs]
    lines.append("Bạn có thể mở từng tin để xem mô tả và ứng tuyển.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


async def answer(message: str, search: SearchClient) -> Reply:
    found = I.classify(message)
    if found.name != I.JOB_SEARCH:
        return Reply(TEXT[found.name])
    if not found.has_criteria:
        return Reply(NO_CRITERIA)
    try:
        jobs = await search.find_jobs(found)
    except SearchUnavailable as exc:
        # no message text in the log: it may contain what the user typed
        log.warning("chat job search failed (%s)", exc.code)
        return Reply(SEARCH_FAILED, [], {"code": SEARCH_FAILED_CODE, "message": SEARCH_FAILED})
    return Reply(_job_text(found, jobs), jobs)
