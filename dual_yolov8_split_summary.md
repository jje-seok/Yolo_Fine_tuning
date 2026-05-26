# Split YOLOv8 구조 요약

이 프로젝트의 최종 모델은 하나의 YOLOv8n을 다음 3개 ONNX/DXNN 파일로 나눈 구조이다.

```text
image
  └─ shared_backbone
        ├─ p3: [1, 64, 80, 80]
        ├─ p4: [1, 128, 40, 40]
        └─ p5: [1, 256, 20, 20]

p3, p4, p5
  ├─ front_head -> front_output: [1, 5, 8400]
  └─ top_head   -> top_output:   [1, 5, 8400]
```

## 역할

- `shared_backbone`: 입력 이미지를 공통 feature `p3/p4/p5`로 변환한다.
- `front_head`: 정면 카메라용 detection head이다.
- `top_head`: 상단/CCTV 카메라용 detection head이다.

## 추론 순서

정면 카메라:

```text
front_image -> shared_backbone -> p3,p4,p5 -> front_head -> front_output
```

상단 카메라:

```text
top_image -> shared_backbone -> p3,p4,p5 -> top_head -> top_output
```

## 출력 의미

각 head의 출력 shape은 `[1, 5, 8400]`이다.

```text
[center_x, center_y, width, height, person_confidence]
```

즉, 두 head 모두 `person` 1개 class만 검출한다.

## 핵심 포인트

- backbone weight는 하나만 공유한다.
- head는 front/top 시점에 맞게 각각 fine-tuning된다.
- DEEPX 컴파일 결과도 3개로 분리된다.

```text
shared_backbone.dxnn
front_head.dxnn
top_head.dxnn
```
