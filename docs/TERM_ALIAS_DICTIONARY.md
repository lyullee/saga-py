# 현장 표현·비표준 용어 사전

term_aliases.json은 HRS/가스안전 질의 라우팅과 혼합검색용 구조화 사전입니다.

현재 223개 표준어 항목과 862개 별칭·약어·현장 표현을 수록했습니다.

## 검색에 쓰는 방식

- expand=true: 표준어와 별칭을 함께 검색어로 확장합니다. 예를 들어 가스통, 봄베, 실린더는 용기 검색 후보를 추가합니다.
- relation=field: 현장식 표현 또는 구어체입니다.
- relation=abbrev: 법령·기술 문서의 약칭입니다.
- relation=typo: PDF OCR 오류나 자주 쓰는 오타입니다.
- relation=related: 서로 연관되지만 법적 정의가 같다고 볼 수 없어 자동 확장하지 않습니다.
- relation=ambiguous: 의미가 여러 개라서 시설 종류·작동 방식·기준번호를 먼저 묻습니다.

안전밸브와 압력조정기, 정압기, 긴급차단장치처럼 이름이 비슷해도 기능·적용 범위가 다른 항목은 일부러 같은 동의어 그룹으로 합치지 않았습니다. 국가법령정보센터의 고압가스 법령 정의에서도 가스설비·고압가스설비·처리설비·감압설비·방호벽·충전설비·압축가스설비가 별도 개념으로 구분됩니다.

## 수집 근거

- [한국가스안전공사 가스제품검사·공장심사](https://www.kgs.or.kr/kgs/afcd/view.do): 용기, 밸브, 안전밸브, 저장탱크, 압력용기, 압축기, 긴급차단장치 등의 공식 명칭
- [국가법령정보센터 고압가스 안전관리법 3단비교표 PDF](https://law.go.kr/lbook/lbFileDownload.do?flExt=pdf&lbookConflSeq=107877&lbookSeq=107471): 법정 정의와 설비 구분
- [국가법령정보센터 액화석유가스 안전관리 및 사업법 시행규칙](https://www.law.go.kr/LSW/lsPdfPrint.do?ancYnChk=0&bylChaChk=N&efGubun=Y&efYd=20231010&joAllCheck=Y&lsiSeq=255297): 저장탱크·배관·비파괴시험·방호벽 공정 용어
- [국가기술표준원 KS 개요](https://kats.go.kr/content.do?cmsid=27): 용어·기술·단위·시험·검사 표준 분류
- [한국가스안전공사 SMS 안내](https://www.kgs.or.kr/kgs/afae/view.do): 안전성향상계획, 내압시험, 기밀시험, 방폭지역구분도 용어
- [국가법령정보센터 수소법 제2조](https://www.law.go.kr/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1026016929): 수소연료공급시설 법정 정의
- [한국가스안전공사 고압가스시설검사](https://www.kgs.or.kr/kgs/afaa/view.do): 중간검사·완성검사·내압·기밀·비파괴검사 용어
- [한국가스안전공사 수소충전소](https://www.kgs.or.kr/hst/fcb/view.do): 수소충전소·압축기·디스펜서 공식 설비 명칭
- [국가법령정보센터 고압가스법 시행규칙 별지 서식](https://www.law.go.kr/LSW/lbook/lbFileDownload.do?flExt=pdf&lbookConflSeq=66103&lbookSeq=71593): 기화장치·독성가스배관용 밸브·실린더캐비닛·잔류가스회수장치 명칭
- [국가법령정보센터 고압가스법 시행규칙 검사기준 별표](https://www.law.go.kr/LSW/flDownload.do?bylClsCd=110201&flSeq=115991739&gubun=): 검사 종류와 안전장치·기밀·내압·비파괴검사 용어
- [국립국어원 한국어기초사전](https://krdict.korean.go.kr/kor/dicMarinerSearch/search): 표준 표제어·관련어·용례와 비속어·구어체 표지 확인

웹에서 찾은 표현은 법적 동의어로 단정하지 않고 field·related·ambiguous 관계를 나눠 보수적으로 검색에 반영합니다.
