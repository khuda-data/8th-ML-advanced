# 이 파일은 cv 디렉토리를 파이썬 패키지로 만들어줍니다.

# cv 패키지에서 외부로 노출할 클래스나 함수를 여기에 정의합니다.
# tracker 서브패키지에서 YOLOv8Tracker 클래스를 가져와서
# cv.YOLOv8Tracker 로 바로 접근할 수 있게 해줍니다.
from .tracker import YOLOv8Tracker 