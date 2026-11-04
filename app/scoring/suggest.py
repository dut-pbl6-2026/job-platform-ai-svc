"""Template suggestions from computed gaps (no LLM). Language follows the CV."""
from app.scoring.specs import CvSpec, JobSpec

_T = {
    "en": {
        "missing": "The job asks for {skills}, but your CV does not show it. If you have this experience, add it with a concrete example; if not, consider a short course or project.",
        "negated": "Your CV says you lack {skills}, which this job lists as a requirement. Be aware this will weigh against the match.",
        "years": "The job asks for about {req:g} years of experience; your CV shows about {have:g}. Make start/end dates explicit for each role so your full experience is counted.",
        "edu": "The job expects a {req} degree. If you hold it, state the degree and institution clearly in an Education section.",
        "semantic": "Your CV text overlaps little with the job description. Tailor the summary and the first bullets of each role to the job title '{title}'.",
        "no_skills_section": "Add a dedicated Skills section listing the tools and technologies you use.",
        "numbers": "Add measurable results (for example 'reduced load time by 30%', 'led a team of 5').",
        "hidden": "Hidden or invisible text was found in the file and ignored. Remove it; it can be read as keyword stuffing.",
        "not_cv": "The file does not look like a CV (no skills, experience or education sections were found).",
    },
    "vi": {
        "missing": "Công việc yêu cầu {skills} nhưng CV chưa thể hiện. Nếu bạn có kinh nghiệm này, hãy bổ sung kèm ví dụ cụ thể; nếu chưa, có thể học một khóa ngắn hoặc làm một dự án nhỏ.",
        "negated": "CV của bạn ghi chưa có {skills}, trong khi công việc yêu cầu. Điều này sẽ làm giảm mức phù hợp.",
        "years": "Công việc yêu cầu khoảng {req:g} năm kinh nghiệm; CV của bạn thể hiện khoảng {have:g} năm. Hãy ghi rõ thời gian bắt đầu/kết thúc ở mỗi vị trí để kinh nghiệm được tính đầy đủ.",
        "edu": "Công việc yêu cầu trình độ {req}. Nếu bạn đã đạt, hãy ghi rõ bằng cấp và trường trong mục Học vấn.",
        "semantic": "Nội dung CV ít liên quan đến mô tả công việc. Hãy điều chỉnh phần tóm tắt và các gạch đầu dòng đầu tiên của mỗi vị trí theo vị trí '{title}'.",
        "no_skills_section": "Thêm mục Kỹ năng liệt kê công cụ và công nghệ bạn sử dụng.",
        "numbers": "Thêm kết quả định lượng (ví dụ 'giảm 30% thời gian tải', 'dẫn dắt nhóm 5 người').",
        "hidden": "Phát hiện văn bản ẩn trong file và đã bỏ qua. Hãy xóa vì có thể bị xem là nhồi từ khóa.",
        "not_cv": "File này không giống một CV (không có mục kỹ năng, kinh nghiệm hay học vấn).",
    },
}
_LEVEL_VI = {"diploma": "cao đẳng", "bachelor": "đại học", "master": "thạc sĩ", "phd": "tiến sĩ"}


def make_suggestions(cv: CvSpec, job: JobSpec, missing: list[str], negated: list[str], comps: dict, caps: list[str], lang: str | None = None) -> list[str]:
    lang = lang if lang in _T else ("vi" if cv.language == "vi" else "en")
    t = _T[lang]
    out: list[str] = []
    if "not_cv_like" in caps:
        out.append(t["not_cv"])
    if negated:
        out.append(t["negated"].format(skills=", ".join(negated[:4])))
    if missing:
        out.append(t["missing"].format(skills=", ".join(missing[:5])))
    if job.required_years and cv.years < job.required_years:
        out.append(t["years"].format(req=job.required_years, have=cv.years))
    ed = comps.get("education")
    if ed is not None and ed < 1.0 and job.required_level != "none":
        out.append(t["edu"].format(req=_LEVEL_VI.get(job.required_level, job.required_level) if lang == "vi" else job.required_level))
    if comps.get("semantic", 1.0) < 0.35:
        out.append(t["semantic"].format(title=job.title))
    if not cv.has_skills_section:
        out.append(t["no_skills_section"])
    if not cv.has_numbers:
        out.append(t["numbers"])
    return out[:6]


def hidden_text_suggestion(language: str) -> str:
    return _T["vi" if language == "vi" else "en"]["hidden"]
