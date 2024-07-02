# Dacon_SaeSSak

## 코드 협업 규칙

<aside>
💡 모든 작업은 다음의 순서로 진행합니다!

1) Issue 발행(Jira and Bitbucket vscode extension에서 연동가능)
2) 발행한 Issue에 해당하는 브랜치를 생성(해결 task 당 하나의 브랜치)
3) 해당 브랜치에서 작업 진행
4) Pull Request + code review
5) develop 브랜치로 합치기

</aside>

<aside>
💡 주의 사항!

1. 작업 전 git pull, 퇴근 시 git push로 항상 최신화 해주기(작업 내용 공유 및 기록 남기기)
2. task 단위(ex. 전처리 작업, 실험, 모듈화)로 Issue와 브랜치를 생성하여 관리하기
3. PR 올라오면 가능한 빨리 처리해주기(다같이…)
4. 하나의 브랜치에선 가능한 한 명이 작업하기
5. 상호 합의할 일 있으면 바로바로 하기
6. **force push 절대금지**

</aside>

### 유형(type) 종류

- `feat`: 새로운 기능
- `exp`: 실험
- `refactor`: 코드 리팩토링
- `docs`: 문서 작업 (README.md, docstring, type hint 등)
- `style`: 코드 포맷팅 관련 작업(pep8, flake8 등)
- `fix`: 버그, 오류 수정
- `chore`: 잡다한 일들. 코드 변경 없음.
- `hotfix`: develop 이상 상위 브랜치에서의 긴급수정. 전체 브랜치 흐름에 영향을 주어선 안됌.

### 브랜치 전략

- Git-flow(main, develop, feature, hotfix 브랜치만 사용)
- 참고자료
    - [Git-flow](https://techblog.woowahan.com/2553/)

### 브랜치 이름 규칙

```bash
<유형(type)>/#<이슈 번호(issue number>-<설명(description)>
```

- 예시 → `feat/#19-add_new_model`

### commit message

- 한 줄 커밋 메시지 작성 시 다음과 같은 포맷을 사용합니다.
    
    ```bash
    <유형(type)>: <설명(description)>
    ```
    
    - 예시 1. 모델의 옵티마이저를 adamw로 바꾸고 beta값을 수정해본 다음 실험을 진행했다면
        
        ```
        feat: change optimizer to AdamW with modified beta values for experiment
        ```
        
    - 예시 2. 코드의 로직(논리 구조)은 큰그림에서 그대로지만 코드를 재구성했을 때
        
        ```
        refactor: refactor code for improved organization and readability
        ```
        
    - 예시 3. README.md와 같은 문서를 수정했을 때
        
        ```
        docs: update README.md
        ```
        

### Issue template(웹으로 하기)

- 제목은 commit message과 동일한 태그를 붙이되 대문자로
- ex. **[FEAT] 학생부 백업 데이터 전처리**


```markdown
## 어떤 기능인가요?

> 추가하려는 기능에 대해 간결하게 설명해주세요

## 작업 상세 내용

- [ ] TODO
- [ ] TODO
- [ ] TODO
```

```markdown
## Details

> 어떤 버그인지 간결하게 설명해주세요

## 어떤 상황에서 발생한 버그인가요?

> (가능하면) Given-When-Then 형식으로 서술해주세요

## 예상 결과

> 예상했던 정상적인 결과가 어떤 것이었는지 설명해주세요

## 참고할만한 자료(선택)
```

### Pull Request template(웹으로 하기)

- 제목은 일반적으로 브랜치명으로 자동 할당
- PR은 bitbucket 웹에서 생성할 경우 템플릿이 등록되어 있으니 웹에서 하길 요망
- {{commit_messages}} → 업데이트 된 커밋 목록이 뜨는 특수활용 문자

```markdown
## 연관된 이슈

> ex) #이슈번호, #이슈번호

## 작업 내용

> 이번 PR에서 작업한 내용을 간략히 설명해주세요

## 리뷰 요구사항(선택)

> 리뷰어가 특별히 봐주었으면 하는 부분이 있다면 작성해주세요
> ex) 메서드 XXX의 이름을 더 잘 짓고 싶은데 혹시 좋은 명칭이 있을까요?

## Relative Commits
{{commit_messages}}
```