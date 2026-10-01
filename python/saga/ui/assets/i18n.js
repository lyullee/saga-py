/* UI labels only. Answers and quoted source documents are localized by the API. */
(() => {
  const key = "saga.language";
  let language = localStorage.getItem(key) === "en" ? "en" : "ko";
  const dictionary = new Map(Object.entries({
    "＋ 새 대화": "＋ New chat",
    "연결 확인 중": "Checking connection",
    "대화 모드": "Conversation mode",
    "기준·법령": "Standards & law",
    "운영·고장": "Operations & faults",
    "사고 사례": "Incident cases",
    "일반 대화": "General chat",
    "내 문서를 검색하고 답변에 근거를 표시합니다.": "Search indexed documents and cite the evidence.",
    "KGS·법령·사규 원문을 우선 검색하고 근거를 표시합니다.": "Search KGS standards, laws, and internal rules with citations.",
    "NREL 운전·고장 통계만 검색해 디지털 트윈 분석에 활용합니다.": "Search NREL operations and failure data for digital-twin analysis.",
    "HIAD 사고 사례만 검색해 원인·결과·대응을 분석합니다.": "Search HIAD cases for causes, consequences, and responses.",
    "문서 검색 없이 선택한 LLM과 자연스럽게 대화합니다.": "Chat with the selected model without document retrieval.",
    "LLM 제공자": "LLM provider",
    "답변 모델": "Answer model",
    "답변 길이": "Answer length",
    "간결하게": "Concise",
    "표준": "Standard",
    "자세히": "Detailed",
    "매우 자세히": "Very detailed",
    "추론 모델": "Model",
    "색인 문서": "Indexed documents",
    "검색 조각": "Search chunks",
    "NREL 소스": "NREL sources",
    "HIAD 소스": "HIAD sources",
    "문서 라이브러리": "Document library",
    "선택한 제공자에만 요청합니다. 문서 검색과 근거 표시는 그대로 유지됩니다.": "Requests use only the selected provider. Document retrieval and citations remain available.",
    "무엇이든 편하게 물어보세요.": "Ask anything about safety.",
    "답변 검증 켜짐": "Answer review enabled",
    "무엇을 확인할까요?": "What would you like to check?",
    "무엇이든 편하게 이야기해 보세요": "What would you like to discuss?",
    "문서 검색부터 비교·절차 분석까지": "Search documents, compare requirements, and analyze procedures",
    "아이디어부터 일상적인 궁금증까지": "From ideas to everyday questions",
    "공개 운전·고장 통계부터 디지털 트윈 상태 해석까지": "From public operations data to digital-twin interpretation",
    "공개 사고 사례부터 원인·결과·재발방지까지": "From incident cases to causes and prevention",
    "RAG 근거 모드 · 색인 문서를 검색하고 인용합니다.": "Evidence mode · Search and cite indexed documents.",
    "기준·법령 RAG 모드 · 색인 문서를 검색하고 인용합니다.": "Standards and law · Search and cite indexed documents.",
    "운영·고장 분석 모드 · NREL 공개 통계를 검색하고 해석합니다.": "Operations and failures · Analyze public NREL data.",
    "사고 사례 분석 모드 · HIAD 공개 사례를 검색하고 해석합니다.": "Incident cases · Analyze public HIAD records.",
    "일반 대화 모드 · 문서 검색 없이 LLM 지식으로 답합니다.": "General chat · Answer without document retrieval.",
    "안전 기준, 절차, 문서 비교를 질문하세요": "Ask about safety standards, procedures, or document comparisons",
    "궁금한 점이나 생각을 편하게 적어보세요": "Ask a question or share a thought",
    "충전량, 충전시간, 압축기 고장, 유지보수 추세를 질문하세요": "Ask about fueling volumes, duration, compressor faults, or maintenance trends",
    "누출·화재·폭발 사고의 원인과 대응을 질문하세요": "Ask about causes and responses to leaks, fires, or explosions",
    "AI 답변은 참고용입니다. 중요한 판단은 반드시 최신 원문과 담당 부서에서 확인하세요.": "AI answers are for reference. Verify important decisions against current source documents and the responsible team.",
    "관리자 토큰 (설정한 경우)": "Admin token (if configured)",
    "PDF 추가": "Add PDF",
    "전체 색인": "Reindex all",
    "법령 API 동기화": "Sync law API",
    "NREL·HIAD 동기화": "Sync NREL & HIAD",
    "문서번호 또는 제목 검색": "Search document code or title",
    "불러오는 중…": "Loading…",
    "문서 목록을 가져오지 못했습니다.": "Could not load documents.",
    "조건에 맞는 문서가 없습니다.": "No matching documents.",
    "공식 원본": "Official source",
    "원문": "Source",
    "보내기": "Send",
    "질문": "Question",
    "대화 모드 선택": "Select conversation mode",
    "LLM 제공자 선택": "Select LLM provider",
    "답변 모델 선택": "Select answer model",
    "답변 길이 선택": "Select answer length",
    "화면 및 답변 언어 선택": "Select interface and answer language",
    "답변을 준비하고 있습니다…": "Preparing an answer…",
    "초안 답변을 작성하고 있습니다…": "Writing a draft answer…",
    "추론형 최종 답변을 정리하고 있습니다…": "Finalizing the reviewed answer…",
    "처리하지 못했습니다.": "Could not complete the request.",
    "응답 데이터가 비어 있습니다": "The response is empty",
    "답변 생성 실패": "Answer generation failed",
    "도움됨": "Helpful",
    "개선 필요": "Needs improvement",
    "의견이 반영되었습니다": "Feedback recorded",
    "근거 데이터": "Evidence",
    "혹시 이 문서를 찾으셨나요?": "Were you looking for this document?",
    "일반 대화 · 문서 검색 안 함": "General chat · No document retrieval",
    "문서 근거 제한 · LLM 보완 설명": "Limited document support · Model explanation",
    "RAG 근거 없음 · LLM 자체 판단": "No RAG evidence · Model judgment",
    "추가 확인 필요": "Further verification needed",
    "NREL 운영·고장 데이터 답변": "NREL operations and failures",
    "HIAD 사고 사례 답변": "HIAD incident cases",
    "문서 근거 답변": "Document-grounded answer",
    "영어 번역은 검증된 한국어 답변을 바탕으로 합니다. 기준 문구는 인용 원문을 확인하세요.": "English rendering is based on the verified Korean answer. Check the original citation text for exact requirements.",
    "PDF를 분석하고 있습니다": "Analyzing PDF",
    "문서 색인이 완료되었습니다": "Document indexing complete",
    "업로드 실패": "Upload failed",
    "변경된 PDF를 색인하고 있습니다": "Indexing changed PDFs",
    "색인 실패": "Indexing failed",
    "NREL·HIAD 공개 데이터를 내려받고 DB화하는 중입니다": "Downloading and indexing public NREL and HIAD data",
    "NREL·HIAD 데이터 색인이 완료되었습니다": "NREL and HIAD indexing complete",
    "공개 데이터 동기화 실패": "Public data sync failed",
    "국가법령정보센터에서 법령을 PDF로 만들고 있습니다": "Creating PDF snapshots from the national law API",
    "법령 동기화 실패": "Law sync failed",
    "일반 대화 모드로 전환했습니다": "Switched to general chat",
    "운영·고장 분석 모드로 전환했습니다": "Switched to operations analysis",
    "사고 사례 분석 모드로 전환했습니다": "Switched to incident analysis",
    "기준·법령 RAG 모드로 전환했습니다": "Switched to standards and law",
    "답변이 끝난 뒤 모드를 바꿔 주세요": "Please switch modes after the answer finishes",
  }));
  const originalText = new WeakMap();
  const originalAttributes = new WeakMap();
  const excluded = ".stage-content,.message.user,.citation,code,pre,script,style";

  function translate(value) {
    const source = String(value ?? "");
    const trimmed = source.trim();
    const translated = dictionary.get(trimmed);
    if (!translated) return source;
    return source.replace(trimmed, translated);
  }
  function translateText(node) {
    if (node.parentElement?.closest(excluded)) return;
    const previous = originalText.get(node);
    const current = node.nodeValue;
    const original = previous && current === previous.translated ? previous.original : current;
    const next = language === "en" ? translate(original) : original;
    originalText.set(node, {original, translated: next});
    if (current !== next) node.nodeValue = next;
  }
  function translateElement(element) {
    if (element.matches?.(excluded) || element.closest?.(excluded)) return;
    const previous = originalAttributes.get(element) || {};
    for (const name of ["placeholder", "title", "aria-label"]) {
      if (!element.hasAttribute(name)) continue;
      const current = element.getAttribute(name);
      const saved = previous[name];
      const original = saved && current === saved.translated ? saved.original : current;
      const next = language === "en" ? translate(original) : original;
      previous[name] = {original, translated: next};
      if (current !== next) element.setAttribute(name, next);
    }
    originalAttributes.set(element, previous);
  }
  function visit(root) {
    if (root.nodeType === Node.TEXT_NODE) { translateText(root); return; }
    if (root.nodeType !== Node.ELEMENT_NODE) return;
    translateElement(root);
    if (root.matches?.(excluded) || root.closest?.(excluded)) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.nodeType === Node.TEXT_NODE) translateText(node);
      else translateElement(node);
    }
  }
  function setLanguage(next) {
    language = next === "en" ? "en" : "ko";
    localStorage.setItem(key, language);
    document.documentElement.lang = language;
    visit(document.body);
    document.dispatchEvent(new CustomEvent("saga-language-change", {detail: {language}}));
  }
  window.SagaI18n = {language: () => language, setLanguage, translate};
  document.documentElement.lang = language;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => visit(document.body));
  else visit(document.body);
  new MutationObserver(records => {
    for (const record of records) {
      if (record.type === "characterData") visit(record.target);
      else if (record.type === "attributes") translateElement(record.target);
      else record.addedNodes.forEach(visit);
    }
  }).observe(document.documentElement, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["placeholder", "title", "aria-label"]});
})();
