const savedAnswerLength = (() => {
  try { return localStorage.getItem("saga.answerLength") || "standard"; } catch { return "standard"; }
})();
const savedChatMode = (() => {
  try { return localStorage.getItem("saga.chatMode") || "rag"; } catch { return "rag"; }
})();
const savedKnowledgeMode = (() => {
  try { return localStorage.getItem("saga.knowledgeMode") || "standards"; } catch { return "standards"; }
})();
const savedProvider = (() => {
  try { return localStorage.getItem("saga.provider") === "groq" ? "groq" : "service_hub"; } catch { return "service_hub"; }
})();
const state = {
  conversationId: null,
  busy: false,
  documents: [],
  models: [],
  model: null,
  provider: savedProvider,
  language: window.SagaI18n?.language() || "ko",
  answerLength: savedAnswerLength,
  chatMode: savedChatMode === "chat" ? "chat" : "rag",
  knowledgeMode: ["standards", "operations", "incidents"].includes(savedKnowledgeMode) ? savedKnowledgeMode : "standards",
};
const $ = selector => document.querySelector(selector);
const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[ch]));

const WELCOME_PROMPTS = Object.freeze([
  "수소 저장탱크와 건축물 사이의 안전거리를 알려줘",
  "수소충전소의 일반적인 위험요인을 쉽게 설명해줘",
  "KGS FS551의 적용 범위를 한눈에 정리해줘",
  "KGS FP217과 FP216은 어떤 점이 다른가요?",
  "수소 저장식 충전시설의 주요 검사 절차를 알려줘",
  "수소 제조식 충전시설에서 꼭 확인할 항목은 무엇인가요?",
  "수소충전소 압력설비의 안전관리 포인트를 알려줘",
  "저장탱크 주변 방호벽 설치 기준을 찾아줘",
  "수소 저장탱크와 도로의 이격거리 기준을 알려줘",
  "충전소 사업소 경계와 설비 사이의 거리를 알려줘",
  "수소충전소와 보호시설 사이의 안전거리를 설명해줘",
  "저장탱크와 인접한 건축물의 배치 시 주의사항은 무엇인가요?",
  "수소충전소 배관의 기본 시설기준을 요약해줘",
  "수소 배관의 압력등급은 어떻게 구분하나요?",
  "고압 수소배관의 누출 위험을 줄이는 방법을 알려줘",
  "수소 배관에서 밸브 설치 위치를 정할 때 고려할 점은 무엇인가요?",
  "수소충전소 배관의 지하 매설 시 확인할 사항을 알려줘",
  "수소 배관의 방호조치 기준을 찾아줘",
  "수소 설비의 비상차단장치 설치 기준을 알려줘",
  "가스누출자동차단장치는 어디에 설치해야 하나요?",
  "가스누출경보기의 설치 위치와 작동 기준을 설명해줘",
  "수소충전소에 필요한 환기 설비 기준을 정리해줘",
  "수소충전소의 통풍 방향을 설계할 때 주의할 점은 무엇인가요?",
  "가연성가스가 체류하기 쉬운 장소를 알려줘",
  "수소 누출 시 폭발 위험이 커지는 조건을 설명해줘",
  "수소의 확산 특성이 안전설계에 어떤 영향을 주나요?",
  "수소충전소 화재 시 초기 대응 절차를 알려줘",
  "수소 누출이 의심될 때 현장에서 먼저 해야 할 조치는 무엇인가요?",
  "충전 중 비상정지 버튼을 눌러야 하는 상황을 알려줘",
  "수소충전소 사고 시 대피구역을 어떻게 설정하나요?",
  "수소 설비의 정전 상황 대응 절차를 정리해줘",
  "수소충전소 화재 예방을 위한 일상 점검 항목을 만들어줘",
  "KGS FS551의 정기검사 주기를 알려줘",
  "KGS FS551의 수시검사는 어떤 경우에 하나요?",
  "KGS FS551 검사방법을 단계별로 정리해줘",
  "배관 시공감리에서 확인하는 항목을 알려줘",
  "배관 정기검사와 수시검사의 차이를 비교해줘",
  "가스시설 완성검사와 정기검사는 무엇이 다른가요?",
  "수소충전소 검사 준비서류를 정리해줘",
  "검사 전에 설비 담당자가 준비하면 좋은 체크리스트를 만들어줘",
  "검사 결과 부적합이 나왔을 때 보완 절차를 알려줘",
  "정기검사에서 누락하기 쉬운 항목을 알려줘",
  "수소 설비 자체점검과 법정검사의 차이를 설명해줘",
  "수소충전소 점검 기록을 어떤 형식으로 남기면 좋을까요?",
  "검사 항목별로 담당자를 배정하는 방법을 제안해줘",
  "수소 배관 기밀시험의 목적과 절차를 알려줘",
  "기밀시험과 누출검사는 어떻게 다른가요?",
  "수소 배관 내압시험의 압력 기준을 찾아줘",
  "내압시험에서 압력 유지시간은 어떻게 확인하나요?",
  "수소 배관 내압시험을 생략할 수 있는 경우가 있나요?",
  "기밀시험에 사용하는 시험매체와 주의사항을 알려줘",
  "기밀시험 중 압력이 떨어지면 어떻게 판단하나요?",
  "수소 설비 누출검지기의 점검 방법을 알려줘",
  "보링 누출검사는 어떤 상황에서 실시하나요?",
  "압력시험 중 이상변형이 발생하면 어떤 조치를 해야 하나요?",
  "수소 압축기 주변의 누출 점검 방법을 정리해줘",
  "수소충전기 호스의 점검 항목을 알려줘",
  "충전 노즐과 리셉터클의 안전점검 방법을 알려줘",
  "수소 디스펜서의 비상정지 기능을 어떻게 시험하나요?",
  "충전설비 접지와 정전기 방지 기준을 설명해줘",
  "수소충전소 전기설비의 방폭구역을 이해하기 쉽게 설명해줘",
  "방폭 전기기기의 점검 시 확인할 표시를 알려줘",
  "수소충전소 조명과 비상전원의 안전기준을 찾아줘",
  "수소충전소 접지저항 측정 절차를 정리해줘",
  "수소 설비의 낙뢰 보호 대책을 알려줘",
  "수소 배관 용접부 검사 기준을 알려줘",
  "용접부 비파괴시험 종류별 특징을 비교해줘",
  "수소 배관 용접 후 외관검사에서 보는 항목은 무엇인가요?",
  "배관 용접 결함이 발견되면 보수 절차를 알려줘",
  "PE 배관 융착 작업 전 확인해야 할 조건을 정리해줘",
  "PE 융착원의 자격과 기록 관리 방법을 알려줘",
  "폴리에틸렌 배관 융착부 검사 기준을 찾아줘",
  "강관과 PE 배관을 연결할 때 주의할 점은 무엇인가요?",
  "수소 배관 재료의 적합성을 어떻게 확인하나요?",
  "배관 자재 성적서에서 확인할 항목을 알려줘",
  "밸브와 플랜지의 재질을 선정할 때 고려할 조건은 무엇인가요?",
  "수소취성에 강한 재료를 선정하는 방법을 설명해줘",
  "수소 배관의 부식 관리 방법을 정리해줘",
  "배관 두께 측정과 잔여수명 평가 방법을 알려줘",
  "배관 외관점검에서 균열과 부식을 구분하는 방법을 알려줘",
  "지하매설 수소배관의 방식 관리 기준을 찾아줘",
  "전기방식 설비의 작동 상태를 어떻게 점검하나요?",
  "배관 관대지전위 측정 결과를 해석하는 방법을 알려줘",
  "수소 배관의 지지대와 신축흡수 조치를 설명해줘",
  "교량에 설치된 가스배관을 점검할 때 무엇을 보나요?",
  "노출 배관의 차량 충돌 방호 기준을 알려줘",
  "배관 표지판과 라인마크 설치 기준을 찾아줘",
  "배관 굴착공사 중 안전관리 절차를 정리해줘",
  "굴착으로 배관이 노출됐을 때 임시 방호 방법을 알려줘",
  "수소 저장탱크의 안전밸브 점검 항목을 알려줘",
  "저장탱크 압력계와 온도계의 점검 주기를 알려줘",
  "저장탱크 과압 방지장치의 역할을 설명해줘",
  "수소 저장용기의 내압시험과 재검사 기준을 알려줘",
  "수소 저장탱크 주변 온도 관리 시 주의할 점은 무엇인가요?",
  "저장탱크의 액면·압력 데이터를 어떻게 모니터링하면 좋을까요?",
  "수소 저장설비의 배수와 방유 대책을 알려줘",
  "수소 저장탱크 기초와 앵커 점검 항목을 정리해줘",
  "저장탱크 주변 차량 진입 방지 대책을 알려줘",
  "수소 저장설비의 점검 동선을 설계하는 방법을 제안해줘",
  "압축기 진동과 이상소음을 점검하는 방법을 알려줘",
  "수소 압축기의 윤활과 냉각 상태를 확인하는 방법은 무엇인가요?",
  "압축가스설비의 과열 위험을 줄이는 방법을 정리해줘",
  "충전 압력 상승이 느릴 때 원인을 추론해줘",
  "충전 중 압력이 갑자기 떨어지는 원인을 알려줘",
  "수소충전소의 충전시간이 길어지는 원인을 점검 순서로 정리해줘",
  "압축기 후단 압력 변동을 어떻게 분석하면 좋을까요?",
  "수소 설비의 PoF와 CoF를 쉽게 설명해줘",
  "수소 배관 위험도 평가를 시작할 때 필요한 데이터는 무엇인가요?",
  "누출 시나리오별 영향범위를 평가하는 방법을 알려줘",
  "수소충전소 위험성평가 회의 진행 순서를 제안해줘",
  "What-if 분석과 HAZOP의 차이를 비교해줘",
  "수소 설비의 안전계장 기능을 점검하는 방법을 알려줘",
  "위험도 매트릭스를 수소충전소에 적용하는 예시를 보여줘",
  "사고예상질문분석(What-if)을 배관 설계에 적용하는 방법을 알려줘",
  "수소충전소 변경관리(MOC) 절차를 정리해줘",
  "설비 변경 후 재검토해야 하는 안전문서를 알려줘",
  "수소충전소 운영자 교육과 자격관리 항목을 제안해줘",
  "교대근무 인수인계서에 꼭 포함할 내용을 만들어줘",
  "수소충전소 비상훈련 시나리오를 3개 만들어줘",
  "누출·화재·정전 상황별 비상연락망 운영 방법을 알려줘",
  "작업허가서에 포함해야 할 수소 작업 안전조건을 알려줘",
  "밀폐공간에서 수소 관련 작업을 할 때 주의할 점은 무엇인가요?",
  "정비 작업 전 LOTO 절차를 수소 설비에 맞게 설명해줘",
  "수소충전소 도급업체 안전관리 체크리스트를 만들어줘",
  "안전관리자의 일일·주간·월간 업무를 나눠서 정리해줘",
  "수소 설비 사고조사 보고서의 기본 구성은 무엇인가요?",
  "사고 원인과 직접 원인을 구분하는 방법을 알려줘",
  "아차사고(near miss) 보고를 활성화하는 방법을 제안해줘",
  "수소충전소 안전성과를 측정할 KPI를 추천해줘",
  "점검 데이터로 반복 고장을 찾는 방법을 알려줘",
  "문서 근거가 부족할 때 담당자에게 확인할 질문을 만들어줘",
  "KGS 기준과 사내 안전관리 절차를 비교하는 표를 만들어줘",
  "두 KGS 기준의 적용범위와 검사주기를 비교해줘",
  "FS551과 FU671의 시설 경계를 설명해줘",
  "FP217과 수소연료사용시설 기준을 함께 적용할 때 주의할 점은 무엇인가요?",
  "같은 설비에 여러 KGS 기준이 적용될 때 우선순위를 어떻게 정하나요?",
  "기준서 개정 전후의 변경사항을 비교하는 방법을 알려줘",
  "문서번호만 알고 있을 때 관련 조항을 빠르게 찾는 방법을 알려줘",
  "특정 조항의 원문과 적용 조건만 간단히 보여줘",
  "이 문서에서 수치가 포함된 조항만 찾아서 표로 정리해줘",
  "문서의 예외 조건만 따로 모아서 설명해줘",
  "안전거리 기준에 영향을 주는 시설 분류를 정리해줘",
  "최신 개정 여부를 확인할 때 어떤 정보를 봐야 하나요?",
  "PDF 원문과 검색 결과가 다를 때 어떻게 검증하나요?",
  "RAG 답변의 인용번호는 어떻게 확인하면 되나요?",
  "문서에 없는 질문을 했을 때 답변이 어떻게 처리되나요?",
  "수소 안전관리와 무관한 질문도 답할 수 있나요?",
  "오늘 서울 날씨를 알려줘",
  "안전관리 회의 시작 전에 사용할 짧은 인사말을 만들어줘",
  "신입 직원에게 수소충전소를 소개하는 쉬운 설명을 써줘",
  "복잡한 안전기준을 경영진 보고용으로 세 문장으로 요약해줘",
  "현장 작업자에게 전달할 친근한 안전 공지문을 작성해줘",
  "수소 안전교육 퀴즈를 5문제 만들어줘",
  "수소충전소 점검 결과를 보고서 문장으로 바꿔줘",
  "기술적인 내용을 비전공자에게 설명하는 방법을 알려줘",
  "안전 점검 회의에서 놓치기 쉬운 질문을 추천해줘",
  "수소 설비 운영 매뉴얼의 목차를 만들어줘",
  "안전관리 업무를 자동화할 수 있는 아이디어를 제안해줘",
  "디지털 트윈으로 수소충전소를 모니터링할 때 필요한 화면을 제안해줘",
  "센서 이상값과 실제 누출을 구분하는 방법을 알려줘",
  "수소충전소 대시보드에 표시할 핵심 지표를 추천해줘",
  "설비 상태 데이터를 이용해 예방정비 일정을 만드는 방법을 알려줘",
  "위험도 변화가 큰 설비를 자동으로 찾아내는 기준을 제안해줘",
  "안전관리 시스템에서 감사 추적성을 확보하는 방법을 알려줘",
]);

const GENERAL_CHAT_PROMPTS = Object.freeze([
  "오늘 해야 할 일을 우선순위로 정리해줘",
  "복잡한 기술 내용을 비전공자도 이해하게 설명해줘",
  "회의에서 사용할 친근한 인사말을 만들어줘",
  "긴 글을 핵심만 세 문단으로 요약하는 방법을 알려줘",
  "새로운 아이디어를 떠올릴 수 있도록 질문을 던져줘",
  "업무 메일을 정중하고 자연스럽게 다듬어줘",
  "어려운 문제를 단계적으로 생각하는 방법을 알려줘",
  "팀 회의 안건과 논의 순서를 제안해줘",
  "한국어 문장을 더 부드럽고 친근하게 바꿔줘",
  "이번 주 자기계발 계획을 현실적으로 짜줘",
  "한 가지 주제를 찬반 관점에서 비교해줘",
  "내가 놓치고 있는 질문이 있는지 함께 점검해줘",
]);

const OPERATIONS_PROMPTS = Object.freeze([
  "NREL 통계에서 수소충전소의 주요 고장 설비를 찾아줘",
  "압축기 유지보수와 고장 추세를 설명해줘",
  "충전소 이용률과 일일 충전량 데이터를 어떻게 해석하나요?",
  "수소 누출 관련 운영 통계가 무엇을 의미하는지 알려줘",
  "NREL 자료를 디지털 트윈 보정에 활용하는 방법을 제안해줘",
  "충전시간과 처리량을 기준으로 상태지표를 만들어줘",
]);

const INCIDENT_PROMPTS = Object.freeze([
  "HIAD에서 수소 누출 사고의 주요 원인을 찾아줘",
  "수소충전소 화재·폭발 사고 사례를 유형별로 정리해줘",
  "사고 사례에서 반복되는 설비와 원인을 설명해줘",
  "HIAD 사고 데이터를 디지털 트윈 시나리오로 바꾸는 방법은?",
  "수소 사고의 직접 원인과 근본 원인을 구분해줘",
  "사고 후 조치와 재발방지 항목을 찾아줘",
]);

const ENGLISH_PROMPTS = Object.freeze({
  standards: [
    "What safety distances apply between hydrogen storage and nearby buildings?",
    "Summarize the scope of KGS FS551 and cite the relevant clauses.",
    "Which inspection steps apply to a hydrogen refueling station?",
    "Compare the requirements in KGS FP217 and FP216.",
    "What should be checked before restarting a hydrogen compressor?",
    "Explain emergency shutoff requirements with source citations.",
  ],
  operations: [
    "What equipment faults are most common in the available NREL station data?",
    "Explain compressor maintenance and failure trends from NREL records.",
    "How should daily fueling volume and utilization be interpreted?",
  ],
  incidents: [
    "Find hydrogen leak incidents in HIAD and summarize recurring causes.",
    "Compare hydrogen fueling-station fire and explosion cases.",
    "What corrective actions recur in the available incident records?",
  ],
  chat: [
    "Explain a complex technical topic for a non-specialist.",
    "Help me organize today's priorities.",
    "Summarize a long document in three short paragraphs.",
  ],
});

function shuffle(items) {
  const result = [...items];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

function renderWelcomePrompts() {
  const container = $("#welcome-prompts");
  if (!container) return;
  const prompts = state.language === "en"
    ? ENGLISH_PROMPTS[state.chatMode === "chat" ? "chat" : state.knowledgeMode]
    : state.chatMode === "chat"
    ? GENERAL_CHAT_PROMPTS
    : state.knowledgeMode === "operations"
      ? OPERATIONS_PROMPTS
      : state.knowledgeMode === "incidents"
        ? INCIDENT_PROMPTS
        : WELCOME_PROMPTS;
  container.innerHTML = shuffle(prompts).slice(0, 3)
    .map(prompt => `<button type="button">${escapeHtml(prompt)}</button>`).join("");
  container.querySelectorAll("button").forEach(button => {
    button.onclick = () => ask(button.textContent);
  });
}

function syncChatModeUI() {
  const isChat = state.chatMode === "chat";
  document.querySelectorAll("[data-knowledge-mode]").forEach(button => {
    const active = button.dataset.knowledgeMode === (isChat ? "chat" : state.knowledgeMode);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const help = $("#mode-help");
  const modeHelp = {
    standards: "KGS·법령·사규 원문을 우선 검색하고 근거를 표시합니다.",
    operations: "NREL 운전·고장 통계만 검색해 디지털 트윈 분석에 활용합니다.",
    incidents: "HIAD 사고 사례만 검색해 원인·결과·대응을 분석합니다.",
  };
  if (help) help.textContent = isChat
    ? "문서 검색 없이 선택한 LLM과 자연스럽게 대화합니다."
    : (modeHelp[state.knowledgeMode] || modeHelp.standards);
  const subtitle = $("#header-subtitle");
  if (subtitle) subtitle.textContent = isChat
    ? "일반 대화 모드 · 문서 검색 없이 LLM 지식으로 답합니다."
    : state.knowledgeMode === "operations"
      ? "운영·고장 분석 모드 · NREL 공개 통계를 검색하고 해석합니다."
      : state.knowledgeMode === "incidents"
        ? "사고 사례 분석 모드 · HIAD 공개 사례를 검색하고 해석합니다."
        : "기준·법령 RAG 모드 · 색인 문서를 검색하고 인용합니다.";
  const message = $("#message");
  if (message) message.placeholder = isChat
    ? "궁금한 점이나 생각을 편하게 적어보세요"
    : state.knowledgeMode === "operations"
      ? "충전량, 충전시간, 압축기 고장, 유지보수 추세를 질문하세요"
      : state.knowledgeMode === "incidents"
        ? "누출·화재·폭발 사고의 원인과 대응을 질문하세요"
        : "안전 기준, 절차, 문서 비교를 질문하세요";
  const welcomeSubtitle = $("#welcome-subtitle");
  if (welcomeSubtitle) welcomeSubtitle.textContent = isChat
    ? "아이디어부터 일상적인 궁금증까지"
    : state.knowledgeMode === "operations"
      ? "공개 운전·고장 통계부터 디지털 트윈 상태 해석까지"
      : state.knowledgeMode === "incidents"
        ? "공개 사고 사례부터 원인·결과·재발방지까지"
        : "문서 검색부터 비교·절차 분석까지";
}

function setupChatMode() {
  syncChatModeUI();
  document.querySelectorAll("[data-knowledge-mode]").forEach(button => {
    button.onclick = () => {
      const nextKnowledgeMode = button.dataset.knowledgeMode;
      const nextMode = nextKnowledgeMode === "chat" ? "chat" : "rag";
      if (nextMode === state.chatMode && (nextMode === "chat" || nextKnowledgeMode === state.knowledgeMode)) return;
      if (state.busy) {
        toast("답변이 끝난 뒤 모드를 바꿔 주세요");
        return;
      }
      state.chatMode = nextMode;
      if (nextMode === "rag") state.knowledgeMode = ["standards", "operations", "incidents"].includes(nextKnowledgeMode) ? nextKnowledgeMode : "standards";
      state.conversationId = null;
      try {
        localStorage.setItem("saga.chatMode", state.chatMode);
        localStorage.setItem("saga.knowledgeMode", state.knowledgeMode);
      } catch { /* ignore storage restrictions */ }
      syncChatModeUI();
      resetWelcome(nextMode === "chat" ? "무엇이든 편하게 이야기해 보세요" : "무엇을 확인할까요?");
      const modeToast = nextMode === "chat" ? "일반 대화 모드로 전환했습니다" : nextKnowledgeMode === "operations" ? "운영·고장 분석 모드로 전환했습니다" : nextKnowledgeMode === "incidents" ? "사고 사례 분석 모드로 전환했습니다" : "기준·법령 RAG 모드로 전환했습니다";
      toast(modeToast);
      $("#message")?.focus();
    };
  });
}

function resetWelcome(title = "무엇을 확인할까요?") {
  const subtitle = state.chatMode === "chat"
    ? "아이디어부터 일상적인 궁금증까지"
    : state.knowledgeMode === "operations"
      ? "공개 운전·고장 통계부터 디지털 트윈 상태 해석까지"
      : state.knowledgeMode === "incidents"
        ? "공개 사고 사례부터 원인·결과·재발방지까지"
        : "문서 검색부터 비교·절차 분석까지";
  $("#conversation").innerHTML = `<div id="welcome" class="welcome"><p id="welcome-subtitle">${subtitle}</p><h2>${escapeHtml(title)}</h2><div id="welcome-prompts" class="prompts" aria-live="polite"></div></div>`;
  renderWelcomePrompts();
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  setTimeout(() => element.classList.remove("show"), 2600);
}

function renderInlineMarkdown(text) {
  let value = escapeHtml(text);
  const codeTokens = [];
  value = value.replace(/`([^`\n]+)`/g, (_, code) => {
    const token = `\u0000CODE${codeTokens.length}\u0000`;
    codeTokens.push(`<code>${code}</code>`);
    return token;
  });
  value = value.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  value = value.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  value = value.replace(/__(.+?)__/g, "<strong>$1</strong>");
  value = value.replace(/~~(.+?)~~/g, "<del>$1</del>");
  value = value.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  value = value.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
  return value.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => codeTokens[Number(index)]);
}

function isTableDivider(line) {
  const cells = line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|");
  return cells.length > 0 && cells.every(cell => /^\s*:?-{3,}:?\s*$/.test(cell));
}

function tableRow(line, tag) {
  const cells = line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|");
  return `<tr>${cells.map(cell => `<${tag}>${renderInlineMarkdown(cell.trim())}</${tag}>`).join("")}</tr>`;
}

function normalizeBrokenEmphasisSections(text) {
  const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  let changed = false;
  let index = 0;
  while (index < lines.length) {
    const marker = lines[index].match(/^\s*\*\*(\d+[.)])\s*$/);
    if (!marker) { index += 1; continue; }
    let next = index + 1;
    while (next < lines.length && !lines[next].trim()) next += 1;
    if (next >= lines.length) { index += 1; continue; }
    let title = lines[next].trim();
    if (!title || /^(?:\*\*)?\d+[.)](?:\s|$)/.test(title)) { index += 1; continue; }
    if (title.endsWith("**")) title = title.slice(0, -2).trim();
    if (!title) { index += 1; continue; }
    lines[index] = `### ${marker[1]} ${title}`;
    lines.splice(index + 1, next - index);
    changed = true;
  }
  return changed ? lines.join("\n") : text;
}

function normalizeBrokenOrderedItems(text) {
  const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  let changed = false;
  let index = 0;
  while (index < lines.length) {
    const marker = lines[index].match(/^(\s*\d+[.)])\s*$/);
    if (!marker) { index += 1; continue; }
    let next = index + 1;
    while (next < lines.length && !lines[next].trim()) next += 1;
    if (next >= lines.length) { index += 1; continue; }
    const nextLine = lines[next].trim();
    if (/^\d+[.)](?:\s|$)/.test(nextLine) || /^[-*+]\s+/.test(nextLine) || /^#{1,6}\s+/.test(nextLine)) {
      index += 1;
      continue;
    }
    lines[index] = `${marker[1]} ${nextLine}`;
    lines.splice(index + 1, next - index);
    changed = true;
  }
  return changed ? lines.join("\n") : text;
}

function normalizeRepeatedSectionLabels(text) {
  const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  let sectionNumber = 0;
  let changed = false;
  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index].match(/^(\s*)(\d+)([.)])\s+(.+?)\s*$/);
    if (!match || match[1]) continue;
    const title = match[4].trim();
    if (title.length > 55 || /\[\d+\]/.test(title)) continue;
    let next = index + 1;
    while (next < lines.length && !lines[next].trim()) next += 1;
    if (next >= lines.length || !/^\s*[-*+]\s+/.test(lines[next])) continue;
    sectionNumber += 1;
    const replacement = `${match[1]}${sectionNumber}${match[3]} ${title}`;
    if (replacement !== lines[index]) {
      lines[index] = replacement;
      changed = true;
    }
  }
  return changed ? lines.join("\n") : text;
}

function formatAnswer(text) {
  const lines = String(normalizeRepeatedSectionLabels(normalizeBrokenOrderedItems(normalizeBrokenEmphasisSections(text))) ?? "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let index = 0;
  let listType = null;
  const closeList = () => {
    if (listType) html.push(`</${listType}>`);
    listType = null;
  };
  while (index < lines.length) {
    const line = lines[index];
    if (/^\s*```/.test(line)) {
      closeList();
      const language = line.replace(/^\s*```/, "").trim();
      const code = [];
      index += 1;
      while (index < lines.length && !/^\s*```/.test(lines[index])) code.push(lines[index++]);
      if (index < lines.length) index += 1;
      const className = language ? ` class="language-${escapeHtml(language)}"` : "";
      html.push(`<pre><code${className}>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    if (!line.trim()) {
      closeList();
      index += 1;
      continue;
    }
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }
    if (/^\s*(?:---+|\*\s*\*\s*\*|___+)\s*$/.test(line)) {
      closeList();
      html.push("<hr>");
      index += 1;
      continue;
    }
    if (index + 1 < lines.length && line.includes("|") && isTableDivider(lines[index + 1])) {
      closeList();
      const rows = [`<table><thead>${tableRow(line, "th")}</thead><tbody>`];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(tableRow(lines[index], "td"));
        index += 1;
      }
      rows.push("</tbody></table>");
      html.push(rows.join(""));
      continue;
    }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      const nextType = ordered ? "ol" : "ul";
      if (listType !== nextType) {
        closeList();
        listType = nextType;
        html.push(`<${listType}>`);
      }
      html.push(`<li>${renderInlineMarkdown((ordered || unordered)[1])}</li>`);
      index += 1;
      continue;
    }
    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) {
      closeList();
      html.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      index += 1;
      continue;
    }
    closeList();
    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() &&
      !/^\s*(?:```|#{1,6}\s|[-*+]\s+|\d+[.)]\s+|>\s?)/.test(lines[index])) {
      paragraph.push(lines[index++]);
    }
    html.push(`<p>${renderInlineMarkdown(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`);
  }
  closeList();
  return html.join("");
}

function messageNode(role, content = "", loading = false) {
  const wrap = document.createElement("article");
  wrap.className = `message ${role}`;
  wrap.innerHTML = `<div class="avatar">${role === "assistant" ? "S" : "나"}</div><div class="bubble">${loading ? '<div class="loading"><i></i><i></i><i></i></div>' : formatAnswer(content)}</div>`;
  $("#conversation").appendChild(wrap);
  wrap.scrollIntoView({behavior: "smooth", block: "end"});
  return wrap;
}

function processingMarkup(text) {
  return `<div class="processing-indicator" aria-live="polite"><span class="processing-icon" aria-hidden="true">◌</span><span class="status-spinner processing-spinner"></span><span class="processing-label">${escapeHtml(text)}</span></div>`;
}

function statusMarkup(text) {
  return processingMarkup(text);
}

function ensureAnswerStages(bubble) {
  if (!bubble.querySelector(".answer-stage")) {
    bubble.innerHTML = `
      <section class="answer-stage draft-stage">
        <div class="stage-label"><span class="stage-dot"></span>초안 답변 · 실시간 생성</div>
        <div class="stage-content draft-content"></div>
      </section>
      <section class="answer-stage final-stage" hidden>
        <div class="stage-label final-label"><span class="stage-dot"></span>추론형 최종 정리</div>
        <div class="stage-content final-content"></div>
      </section>`;
  }
  return {
    draft: bubble.querySelector(".draft-content"),
    final: bubble.querySelector(".final-content"),
    finalStage: bubble.querySelector(".final-stage"),
  };
}

function renderDraftAnswer(pending, text) {
  const bubble = pending.querySelector(".bubble");
  if (!bubble) return;
  const stages = ensureAnswerStages(bubble);
  stages.draft.innerHTML = formatAnswer(text || "");
  pending.dataset.draftRendered = "true";
}

function renderFinalAnswer(pending, text) {
  const bubble = pending.querySelector(".bubble");
  if (!bubble) return;
  const stages = ensureAnswerStages(bubble);
  if (!(pending.streamText || pending.draftAnswer)) {
    bubble.querySelector(".draft-stage")?.remove();
  }
  stages.final.innerHTML = formatAnswer(text || "");
  stages.finalStage.hidden = false;
  pending.dataset.finalRendered = "true";
}

function updateProcessingStatus(pending, text) {
  const bubble = pending.querySelector(".bubble");
  if (!bubble) return;
  const indicator = bubble.querySelector(".processing-indicator");
  if (indicator) {
    const label = indicator.querySelector(".processing-label");
    if (label) label.textContent = text || "답변을 준비하고 있습니다…";
    return;
  }
  if (pending.streamText || pending.dataset.draftStarted === "true") {
    bubble.insertAdjacentHTML("beforeend", processingMarkup(text || "답변을 준비하고 있습니다…"));
  } else {
    bubble.innerHTML = processingMarkup(text || "답변을 준비하고 있습니다…");
  }
}

function answerModeMarkup(mode, answer = "", requestedMode = state.chatMode) {
  if (requestedMode === "chat") {
    return '<span class="answer-mode chat-only">일반 대화 · 문서 검색 안 함</span>';
  }
  if (mode === "llm_only" && answer.includes("검색된 문서가 질문의 핵심")) {
    return '<span class="answer-mode llm-only">문서 근거 제한 · LLM 보완 설명</span>';
  }
  if (mode === "llm_only") return '<span class="answer-mode llm-only">RAG 근거 없음 · LLM 자체 판단</span>';
  if (mode === "clarification") return '<span class="answer-mode clarification">추가 확인 필요</span>';
  if (state.knowledgeMode === "operations") return '<span class="answer-mode operations">NREL 운영·고장 데이터 답변</span>';
  if (state.knowledgeMode === "incidents") return '<span class="answer-mode incidents">HIAD 사고 사례 답변</span>';
  return '<span class="answer-mode rag">문서 근거 답변</span>';
}

function renderCitations(citations) {
  if (!citations?.length) return "";
  return `<div class="citations"><div class="citations-title">근거 데이터 · ${citations.length}</div>${citations.map(c => {
    const href = c.source_url || `/files/${c.document_id}`;
    const sourceLabel = c.source_url ? "공식 원본" : `원문 ${c.page}쪽`;
    return `<a class="citation" href="${escapeHtml(href)}" target="_blank" rel="noopener"><span class="citation-number">${c.number}</span><span><strong>${escapeHtml(c.doc_code)} · ${escapeHtml(c.title)}</strong><br>${escapeHtml(c.hierarchy)} · ${sourceLabel}</span></a>`;
  }).join("")}</div>`;
}

function renderSuggestions(suggestions) {
  if (!suggestions?.length) return "";
  return `<div class="suggestions"><div class="suggestions-title">혹시 이 문서를 찾으셨나요?</div>${suggestions.map(item => `<button class="suggestion" data-code="${escapeHtml(item.doc_code)}"><strong>${escapeHtml(item.doc_code)}</strong><span>${escapeHtml(item.title)}</span></button>`).join("")}</div>`;
}

function renderExtras(data) {
  const languageNote = data.language === "en" ? '<p class="translation-note">English rendering is based on the verified Korean answer. Check the original citation text for exact requirements.</p>' : "";
  return `<div class="answer-meta">${answerModeMarkup(data.answer_mode, data.answer, data.mode || state.chatMode)}<span class="answer-model">${escapeHtml(data.model || state.model || "Service Hub")}</span></div>${languageNote}${renderSuggestions(data.suggestions)}${renderCitations(data.citations)}<div class="feedback"><button data-score="1">도움됨</button><button data-score="-1">개선 필요</button></div>`;
}

async function typeAnswer(bubble, text) {
  bubble.classList.add("answer-bubble");
  const step = text.length > 1800 ? 14 : 8;
  for (let index = 0; index < text.length; index += step) {
    bubble.innerHTML = formatAnswer(text.slice(0, index + step));
    bubble.closest(".message")?.scrollIntoView({behavior: "smooth", block: "end"});
    await new Promise(resolve => setTimeout(resolve, 8));
  }
  bubble.innerHTML = formatAnswer(text);
}

async function health() {
  try {
    const data = await fetch("/api/health").then(response => response.json());
    const configured = state.provider === "groq" ? data.groq_configured : data.service_hub_configured;
    $("#status-dot").classList.toggle("ready", Boolean(configured));
    $("#status-text").textContent = configured ? `${state.provider === "groq" ? "Groq" : "Service Hub"} 연결 준비` : "선택한 제공자 API 키 설정 필요";
    $("#model-name").textContent = (state.model || data.model).split("/").pop();
    $("#doc-count").textContent = Number(data.documents || 0).toLocaleString();
    $("#chunk-count").textContent = Number(data.chunks || 0).toLocaleString();
    $("#nrel-count").textContent = Number(data.nrel_documents || 0).toLocaleString();
    $("#hiad-count").textContent = Number(data.hiad_documents || 0).toLocaleString();
  } catch {
    $("#status-text").textContent = "서버 연결 실패";
  }
}

async function loadModels() {
  try {
    const provider = state.provider;
    const data = await fetch(`/api/models?provider=${encodeURIComponent(provider)}`).then(response => response.json());
    if (provider !== state.provider) return;
    state.models = data.models || [];
    state.model = data.selected || state.models[0] || null;
    const select = $("#model-select");
    select.innerHTML = state.models.map(model => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("");
    if (state.model) select.value = state.model;
    select.onchange = event => { state.model = event.target.value; $("#model-name").textContent = state.model; };
    $("#model-name").textContent = state.model || "—";
    health();
  } catch {
    toast("모델 목록을 가져오지 못했습니다");
  }
}

function setupAnswerLength() {
  const select = $("#answer-length-select");
  if (!select) return;
  const allowed = new Set(["concise", "standard", "detailed", "very_detailed"]);
  if (!allowed.has(state.answerLength)) state.answerLength = "standard";
  select.value = state.answerLength;
  select.onchange = event => {
    state.answerLength = allowed.has(event.target.value) ? event.target.value : "standard";
    try { localStorage.setItem("saga.answerLength", state.answerLength); } catch { /* ignore storage restrictions */ }
  };
}

function handleEvent(eventName, data, pending) {
  if (eventName === "status") {
    updateProcessingStatus(pending, data.text || "답변을 준비하고 있습니다…");
    return null;
  }
  if (eventName === "draft_start") {
    pending.dataset.draftStarted = "true";
    updateProcessingStatus(pending, data.text || "초안 답변을 실시간으로 작성하고 있습니다…");
    return null;
  }
  if (eventName === "draft_token" || eventName === "token") {
    const text = String(data.text || "");
    if (!text) return null;
    pending.dataset.tokenStarted = "true";
    pending.streamText = `${pending.streamText || ""}${text}`;
    const bubble = pending.querySelector(".bubble");
    bubble.classList.add("answer-bubble");
    pending.dataset.draftStarted = "true";
    renderDraftAnswer(pending, pending.streamText);
    if (!bubble.querySelector(".processing-indicator")) {
      bubble.insertAdjacentHTML("beforeend", processingMarkup("초안 답변을 작성하고 있습니다…"));
    }
    pending.scrollIntoView({behavior: "smooth", block: "end"});
    return null;
  }
  if (eventName === "final_start") {
    updateProcessingStatus(pending, data.text || "추론형 최종 답변을 정리하고 있습니다…");
    return null;
  }
  if (eventName === "answer") return data;
  if (eventName === "error") throw new Error(data.detail || "답변 생성 실패");
  return null;
}

async function streamAnswer(request, pending) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "답변 생성 실패");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const lines = block.split(/\r?\n/);
      const eventName = (lines.find(line => line.startsWith("event:")) || "event: message").slice(6).trim();
      const dataLine = lines.find(line => line.startsWith("data:"));
      if (!dataLine) continue;
      const eventData = JSON.parse(dataLine.slice(5).trim() || "{}");
      if (eventName === "draft") {
        // Compatibility with older servers.  Even a completed legacy draft
        // is rendered in its own stage and is never replaced by the review.
        pending.dataset.draftStarted = "true";
        pending.draftAnswer = eventData.answer || "";
        pending.streamText = pending.draftAnswer;
        renderDraftAnswer(pending, pending.draftAnswer);
        continue;
      }
      const handled = handleEvent(eventName, eventData, pending);
      if (handled) result = handled;
    }
    if (done) break;
  }
  return result;
}

async function ask(message) {
  if (state.busy || !message.trim()) return;
  state.busy = true;
  $("#welcome")?.remove();
  messageNode("user", message);
  const pending = messageNode("assistant", "", true);
  $("#send").disabled = true;
  try {
    const data = await streamAnswer({message, conversation_id: state.conversationId, model: state.model, provider: state.provider, language: state.language, answer_length: state.answerLength, mode: state.chatMode, knowledge_mode: state.knowledgeMode}, pending);
    if (!data) throw new Error("응답 데이터가 비어 있습니다");
    state.conversationId = data.conversation_id;
    const bubble = pending.querySelector(".bubble");
    bubble.querySelector(".processing-indicator")?.remove();
    const draftText = pending.streamText || pending.draftAnswer || data.draft_answer || "";
    if (draftText) renderDraftAnswer(pending, draftText);
    // Keep the streamed first pass above and append the reviewed answer below.
    // This is intentionally not a typewriter animation: the first pass has
    // already arrived as real SSE deltas, so the final pass should appear as
    // a clearly labelled second stage when validation finishes.
    const finalText = data.final_answer || data.answer || "";
    if (data.language === "en" && draftText && draftText.trim() === finalText.trim()) {
      // The verified English rendering has already arrived as real SSE deltas.
      pending.querySelector(".draft-stage .stage-label").textContent = "English answer · live";
      pending.querySelector(".final-stage")?.remove();
    } else {
      renderFinalAnswer(pending, finalText);
    }
    bubble.insertAdjacentHTML("beforeend", renderExtras({...data, model: state.model, mode: data.mode || state.chatMode}));
    pending.querySelectorAll("[data-code]").forEach(button => button.onclick = () => ask(`KGS ${button.dataset.code}은 무엇인지 핵심 내용과 적용 범위를 알려줘`));
    pending.querySelectorAll("[data-score]").forEach(button => button.onclick = async () => {
      await fetch(`/api/log/${data.log_id}/feedback`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({score:Number(button.dataset.score)})});
      toast("의견이 반영되었습니다");
    });
  } catch (error) {
    pending.querySelector(".bubble").innerHTML = `<strong>처리하지 못했습니다.</strong><br>${escapeHtml(error.message)}`;
  } finally {
    state.busy = false;
    $("#send").disabled = false;
  }
}

$("#chat-form").addEventListener("submit", event => {event.preventDefault(); const box = $("#message"); const value = box.value; box.value = ""; box.style.height = "auto"; ask(value);});
$("#message").addEventListener("keydown", event => {if (event.key === "Enter" && !event.shiftKey) {event.preventDefault(); $("#chat-form").requestSubmit();}});
$("#message").addEventListener("input", event => {event.target.style.height = "auto"; event.target.style.height = `${Math.min(event.target.scrollHeight, 170)}px`;});
$("#new-chat").onclick = () => {state.conversationId = null; resetWelcome(state.chatMode === "chat" ? "무엇이든 편하게 이야기해 보세요" : "무엇을 확인할까요?"); $("#message").focus();};

function toggleLibrary(open) {$("#library").classList.toggle("open", open); $("#backdrop").classList.toggle("open", open); $("#library").setAttribute("aria-hidden", String(!open)); if (open) loadDocuments();}
$("#open-library").onclick = () => toggleLibrary(true); $("#close-library").onclick = () => toggleLibrary(false); $("#backdrop").onclick = () => toggleLibrary(false);
async function loadDocuments() {const list = $("#document-list"); list.innerHTML = "불러오는 중…"; try {state.documents = await fetch("/api/documents").then(r => r.json()); renderDocuments();} catch {list.innerHTML = "문서 목록을 가져오지 못했습니다.";}}
function renderDocuments() {const query = $("#doc-filter").value.toLowerCase(); const items = state.documents.filter(item => `${item.doc_code} ${item.title}`.toLowerCase().includes(query)); $("#document-list").innerHTML = items.map(item => { const href = item.source_url || `/files/${item.id}`; const label = item.source_url ? "공식 원본" : "원문"; return `<div class="document-card"><span class="doc-type">${escapeHtml(item.doc_type)}</span><div><strong>${escapeHtml(item.doc_code)} · ${escapeHtml(item.title)}</strong><p>${item.page_count}쪽 · ${item.chunk_count}개 검색 조각</p></div><a href="${escapeHtml(href)}" target="_blank" rel="noopener">${label}</a></div>`; }).join("") || "조건에 맞는 문서가 없습니다.";}
$("#doc-filter").addEventListener("input", renderDocuments);
function adminHeaders() {const token = $("#admin-token").value; return token ? {"X-Admin-Token": token} : {};}
$("#pdf-upload").onchange = async event => {const file = event.target.files[0]; if (!file) return; const body = new FormData(); body.append("file", file); toast("PDF를 분석하고 있습니다"); const response = await fetch("/api/documents/upload", {method:"POST", headers:adminHeaders(), body}); const data = await response.json(); if (response.ok) {toast("문서 색인이 완료되었습니다"); await health(); await loadDocuments();} else toast(data.detail || "업로드 실패"); event.target.value = "";};
$("#reindex").onclick = async () => {toast("변경된 PDF를 색인하고 있습니다"); const response = await fetch("/api/documents/reindex", {method:"POST", headers:adminHeaders()}); const data = await response.json(); if (response.ok) {toast(`색인 ${data.indexed} · OCR 필요 ${data.needs_ocr} · 유지 ${data.unchanged} · 실패 ${data.failed.length}`); await health(); await loadDocuments();} else toast(data.detail || "색인 실패");};
$("#external-sync").onclick = async () => {const button = $("#external-sync"); button.disabled = true; toast("NREL·HIAD 공개 데이터를 내려받고 DB화하는 중입니다"); try {const response = await fetch("/api/sources/sync", {method:"POST", headers:{...adminHeaders(), "Content-Type":"application/json"}, body:JSON.stringify({sources:["nrel","hiad"], force:false})}); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "공개 데이터 동기화 실패"); const failed = (data.failed || []).length; toast(failed ? `완료 · 실패 ${failed}건` : "NREL·HIAD 데이터 색인이 완료되었습니다"); await health(); await loadDocuments();} catch (error) {toast(error.message || "공개 데이터 동기화 실패");} finally {button.disabled = false;}};
$("#law-sync").onclick = async () => {
  const button = $("#law-sync");
  button.disabled = true;
  toast("국가법령정보센터에서 법령을 PDF로 만들고 있습니다");
  try {
    const response = await fetch("/api/laws/sync", {
      method: "POST",
      headers: {...adminHeaders(), "Content-Type": "application/json"},
      body: JSON.stringify({force: false}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "법령 동기화 실패");
    const failed = (data.failed || []).length;
    toast(failed ? `법령 ${data.indexed}건 색인 · ${failed}건 실패` : `법령 ${data.indexed}건 색인 완료`);
    await health();
    await loadDocuments();
  } catch (error) {
    toast(error.message || "법령 동기화 실패");
  } finally {
    button.disabled = false;
  }
};

$("#provider-select").value = state.provider;
$("#language-select").value = state.language;
$("#language-select").onchange = event => {
  state.language = event.target.value === "en" ? "en" : "ko";
  window.SagaI18n?.setLanguage(state.language);
  syncChatModeUI();
  renderWelcomePrompts();
};
$("#provider-select").onchange = event => {
  state.provider = event.target.value === "groq" ? "groq" : "service_hub";
  try { localStorage.setItem("saga.provider", state.provider); } catch { /* ignore storage restrictions */ }
  loadModels();
};
health();
loadModels();
setupAnswerLength();
setupChatMode();
renderWelcomePrompts();
