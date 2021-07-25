# klue-baseline

> "KLUE로 모델 평가하기" 크루는 KLUE 벤치마크를 경험하는 것이 주 목표입니다.  8개의 과제에 대한 baseline 을 작성하고 KLUE 벤치마크로 평가해보려 합니다.  과연 리더보드에 기록될 수 있을까요 ? 

- 노션 페이지 : [바로가기](https://www.notion.so/chanrankim/KLUE-85d3c31ec1ea46c9b0e7fa39c52c86f9)
- 빌드 페이지 : [https://pseudo-lab.github.io/klue-baseline/](https://pseudo-lab.github.io/klue-baseline/)

## jupyter book build

- init

  - 페이지 작성은 `.md`, `.ipynb` 형식으로 작성
  - cmd 사용
    - anaconda prompt 설치 권장
    - 가상환경에서 설치 권장(옵션)

- git clone 

  - ```
    git clone https://github.com/Pseudo-Lab/klue-baseline.git
    ```

- 페이지 작성 파일 이동

  - `pytorch-guide/book/docs` 에 위치시킬 것
  - `ch1` 폴더 내에 작성

- `_toc.yml` 변경

  - `pytorch-guide/book` 내 `_toc.yml` 파일 변경

  - ```yaml
    format: jb-book
    root: docs/index
    chapters:
    - file: docs/Topic Classification
    # section:(하위 페이지를 작성하고 싶은 경우)
    #  - file: docs/(작성한 파일 이름 작성)
    - file: docs/Semantic Textual Similarity
    - file: docs/Natural language inference
    - file: docs/Named entity recognition
    - file: docs/Relation extraction
    - file: docs/Dependency Parsing
    - file: docs/Machine Reading Comprehension
    - file: docs/Dialogue State Tracking
    ```
    
  - 위 코드의 주석 참조하여 추가한 페이지 이름 변경

- Jupyter book 설치

  - ```
    pip install -U jupyter-book
    ```

- 폴더 이동

  - ```
    cd klue-baseline
    ```

- (로컬) Jupyter book build

  - ```
    jupyter-book build book/
    ```

  - cmd 창 내 `Or paste this line directly into your browser bar:` 이하의 링크를 사용하면 로컬에서 jupyter book 을 빌드할 수 있음

- (온라인) Jupyter book build

  - 변경 내용 push 할 것

  - ```python
    pip install ghp-import
    ghp-import -n -p -f book/_build/html -m "initial publishing"
    ```

  - [https://pseudo-lab.github.io/klue-baseline/](https://pseudo-lab.github.io/klue-baseline/) 링크 접속

