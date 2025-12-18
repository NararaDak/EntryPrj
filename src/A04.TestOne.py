import A04 as a04




class YOLOv8Report(a04.YOLOv8Model):
    """
    YOLOv8 기반 객체 탐지 모델
    - Ultralytics YOLOv8 사용
    - 객체 탐지 및 분류 동시 수행
    """

    def __init__(self, model_size="n", num_classes=None):
        """
        Args:
            model_size: YOLOv8 모델 크기 ('n', 's', 'm', 'l', 'x')
            num_classes: 클래스 수 (필수 파라미터)
        """
        super(YOLOv8Report, self).__init__()
        self.model_size = model_size
        self.num_classes = num_classes
        print(f"YOLOv8Report 모델 초기화 완료: 크기={model_size}, 클래스 수={num_classes}")

    def testModelOneFile(self, pt_file, test_img_dir, test_loader=None):
        """Best 모델 파일로 테스트 및 Submission 생성
        
        Args:
            pt_file: 로드할 .pt 모델 파일 경로
            test_img_dir: 테스트 이미지 디렉토리
            test_loader: 테스트 데이터 로더 (선택, 사용하지 않음)
        """
        OpLog(f"Best 모델로 테스트 시작: {pt_file}", bLines=True)
        
        # 모델 로드
        if not self.load_yolo_model(pt_file):
            OpLog(f"모델 로드 실패: {pt_file}", bLines=True)
            return False
        
        # testModel 호출 (epoch=1, max_epochs=1로 설정)
        self.testModel(test_img_dir, test_loader, epoch=1, max_epochs=1)
        
        OpLog(f"Best 모델 테스트 및 Submission 생성 완료", bLines=True)
        return True

testbest():
    model = YOLOv8Report("n", 74)
    # YOLOv8 모델은 load_yolo_model을 사용해야 합니다
    model.load_yolo_model(a04.MODEL_FILES + "/yolobest.pt")
    test_image_file = r"D:\01.project\EntryPrj\data\oraldrug\test_images\1.png"
    yaml_file = r"D:\01.project\EntryPrj\data\oraldrug\yolo_yaml.yaml"
    model.testModelOneFile(test_image_file, yaml_file, 1, 1)

testbest()
