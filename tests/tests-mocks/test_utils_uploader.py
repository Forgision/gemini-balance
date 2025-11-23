from unittest.mock import patch, MagicMock
from app.utils.uploader import ImageUploaderFactory, SmMsUploader, PicGoUploader, AliyunOSSUploader, CloudFlareImgBedUploader

def test_image_uploader_factory():
    """Test the ImageUploaderFactory."""
    uploader = ImageUploaderFactory.create("smms", api_key="test_key")
    assert isinstance(uploader, SmMsUploader)

    uploader = ImageUploaderFactory.create("picgo", api_key="test_key")
    assert isinstance(uploader, PicGoUploader)

    uploader = ImageUploaderFactory.create("aliyun_oss", access_key="test_key", access_key_secret="test_secret", bucket_name="test_bucket", endpoint="test_endpoint", region="test_region")
    assert isinstance(uploader, AliyunOSSUploader)

    uploader = ImageUploaderFactory.create("cloudflare_imgbed", auth_code="test_code", base_url="http://example.com")
    assert isinstance(uploader, CloudFlareImgBedUploader)

@patch("requests.post")
def test_smms_uploader(mock_post):
    """Test the SmMsUploader."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "data": {"width": 100, "height": 100, "filename": "test.jpg", "size": 123, "url": "http://example.com/test.jpg", "delete": "http://example.com/delete/test.jpg"}}
    mock_post.return_value = mock_response

    uploader = SmMsUploader("test_key")
    response = uploader.upload(b"image_data", "test.jpg")

    assert response.success is True
    assert response.data.url == "http://example.com/test.jpg"

@patch("requests.post")
def test_picgo_uploader(mock_post):
    """Test the PicGoUploader."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status_code": 200, "image": {"width": 100, "height": 100, "filename": "test.jpg", "size": 123, "url": "http://example.com/test.jpg", "delete_url": "http://example.com/delete/test.jpg"}}
    mock_post.return_value = mock_response

    uploader = PicGoUploader("test_key")
    response = uploader.upload(b"image_data", "test.jpg")

    assert response.success is True
    assert response.data.url == "http://example.com/test.jpg"

@patch("requests.put")
def test_aliyun_oss_uploader(mock_put):
    """Test the AliyunOSSUploader."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_put.return_value = mock_response

    uploader = AliyunOSSUploader("test_key", "test_secret", "test_bucket", "test_endpoint", "test_region")
    response = uploader.upload(b"image_data", "test.jpg")

    assert response.success is True
    assert response.data.url == "https://test_bucket.test_endpoint/test.jpg"

@patch("requests.post")
def test_cloudflare_imgbed_uploader(mock_post):
    """Test the CloudFlareImgBedUploader."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"src": "/test.jpg"}]
    mock_post.return_value = mock_response

    uploader = CloudFlareImgBedUploader("test_code", "http://example.com/upload")
    response = uploader.upload(b"image_data", "test.jpg")

    assert response.success is True
    assert response.data.url == "http://example.com/test.jpg"
