import requests
from app.domain.image_models import ImageMetadata, ImageUploader, UploadResponse
from enum import Enum
from typing import Optional, Any
import hashlib
import base64
import hmac
from datetime import datetime
from app.log.logger import get_image_create_logger


class UploadErrorType(Enum):
    """Upload error type enum."""

    NETWORK_ERROR = "network_error"  # Network request error
    AUTH_ERROR = "auth_error"  # Authentication error
    INVALID_FILE = "invalid_file"  # Invalid file
    SERVER_ERROR = "server_error"  # Server error
    PARSE_ERROR = "parse_error"  # Response parsing error
    UNKNOWN = "unknown"  # Unknown error


class UploadError(Exception):
    """Image upload error exception class."""

    def __init__(
        self,
        message: str,
        error_type: UploadErrorType = UploadErrorType.UNKNOWN,
        status_code: Optional[int] = None,
        details: Optional[dict] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize the upload error exception.

        Args:
            message: Error message
            error_type: Error type
            status_code: HTTP status code
            details: Detailed error information
            original_error: Original exception
        """
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}
        self.original_error = original_error

        # Build the full error message
        full_message = f"[{error_type.value}] {message}"
        if status_code:
            full_message = f"{full_message} (Status: {status_code})"
        if details:
            full_message = f"{full_message} - Details: {details}"

        super().__init__(full_message)

    @classmethod
    def from_response(
        cls, response: Any, message: Optional[str] = None
    ) -> "UploadError":
        """
        Create an error instance from an HTTP response.

        Args:
            response: HTTP response object
            message: Custom error message
        """
        try:
            error_data = response.json()
            details = error_data.get("data", {})
            return cls(
                message=message or error_data.get("message", "Unknown error"),
                error_type=UploadErrorType.SERVER_ERROR,
                status_code=response.status_code,
                details=details,
            )
        except Exception:
            return cls(
                message=message or "Failed to parse error response",
                error_type=UploadErrorType.PARSE_ERROR,
                status_code=response.status_code,
            )


class SmMsUploader(ImageUploader):
    API_URL = "https://sm.ms/api/v2/upload"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def upload(self, file: bytes, filename: str) -> UploadResponse:
        try:
            # Prepare request headers
            headers = {"Authorization": f"Basic {self.api_key}"}

            # Prepare file data
            files = {"smfile": (filename, file, "image/png")}

            # Send the request
            response = requests.post(self.API_URL, headers=headers, files=files)

            # Check the response status
            response.raise_for_status()

            # Parse the response
            result = response.json()

            # Verify if the upload was successful
            if not result.get("success"):
                raise UploadError(result.get("message", "Upload failed"))

            # Convert to a unified format
            data = result["data"]
            image_metadata = ImageMetadata(
                width=data["width"],
                height=data["height"],
                filename=data["filename"],
                size=data["size"],
                url=data["url"],
                delete_url=data["delete"],
            )

            return UploadResponse(
                success=True,
                code="success",
                message="Upload success",
                data=image_metadata,
            )

        except requests.RequestException as e:
            # Handle network request related errors
            raise UploadError(f"Upload request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            # Handle response parsing errors
            raise UploadError(f"Invalid response format: {str(e)}")
        except Exception as e:
            # Handle other unexpected errors
            raise UploadError(f"Upload failed: {str(e)}")


class QiniuUploader(ImageUploader):
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key

    def upload(self, file: bytes, filename: str) -> UploadResponse:
        # Implement the specific upload logic for Qiniu
        raise NotImplementedError("QiniuUploader is not implemented yet.")


class PicGoUploader(ImageUploader):
    """Chevereto API Image Uploader"""

    def __init__(
        self, api_key: str, api_url: str = "https://www.picgo.net/api/1/upload"
    ):
        """
        Initialize the Chevereto uploader.

        Args:
            api_key: Chevereto API key
            api_url: Chevereto API upload address
        """
        self.api_key = api_key
        self.api_url = api_url

    def upload(self, file: bytes, filename: str) -> UploadResponse:
        """
        Upload an image to the Chevereto service.

        Args:
            file: Image file binary data
            filename: File name

        Returns:
            UploadResponse: Upload response object

        Raises:
            UploadError: Thrown when the upload fails
        """
        try:
            # Prepare request headers
            headers = {}

            # Build the request URL
            request_url = self.api_url

            # Check if it is the default PicGo URL, if so, use header authentication, otherwise use URL parameter authentication
            if self.api_url == "https://www.picgo.net/api/1/upload":
                headers["X-API-Key"] = self.api_key
            else:
                # For custom URLs, add the API key as a query parameter to the URL
                from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

                parsed_url = urlparse(request_url)
                query_params = parse_qs(parsed_url.query)
                query_params["key"] = [self.api_key]
                new_query = urlencode(query_params, doseq=True)
                request_url = urlunparse(parsed_url._replace(query=new_query))

            # Prepare file data
            files = {"source": (filename, file)}

            # Send the request
            response = requests.post(request_url, headers=headers, files=files)

            # Check the response status
            response.raise_for_status()

            # Parse the response
            result = response.json()

            # Handle the response format of a custom PicGo server
            if "success" in result and "result" in result:
                # Custom PicGo server format: {"success": true, "result": ["url"]}
                if result["success"]:
                    image_url = (
                        result["result"][0]
                        if result["result"] and len(result["result"]) > 0
                        else ""
                    )
                    image_metadata = ImageMetadata(
                        width=0,
                        height=0,
                        filename=filename,
                        size=0,
                        url=image_url,
                        delete_url=None,
                    )
                    return UploadResponse(
                        success=True,
                        code="success",
                        message="Upload success",
                        data=image_metadata,
                    )
                else:
                    raise UploadError(
                        message="Upload failed",
                        error_type=UploadErrorType.SERVER_ERROR,
                        status_code=400,
                        details=result,
                    )

            # Handle the response format of the official PicGo server
            # Verify if the upload was successful
            if result.get("status_code") != 200:
                error_message = "Upload failed"
                if "error" in result:
                    error_message = result["error"].get("message", error_message)
                raise UploadError(
                    message=error_message,
                    error_type=UploadErrorType.SERVER_ERROR,
                    status_code=result.get("status_code"),
                    details=result.get("error"),
                )

            # Extract image information from the response
            image_data = result.get("image", {})

            # Build image metadata
            image_metadata = ImageMetadata(
                width=image_data.get("width", 0),
                height=image_data.get("height", 0),
                filename=image_data.get("filename", filename),
                size=image_data.get("size", 0),
                url=image_data.get("url", ""),
                delete_url=image_data.get("delete_url", None),
            )

            return UploadResponse(
                success=True,
                code="success",
                message=result.get("success", {}).get("message", "Upload success"),
                data=image_metadata,
            )

        except requests.RequestException as e:
            # Handle network request related errors
            raise UploadError(
                message=f"Upload request failed: {str(e)}",
                error_type=UploadErrorType.NETWORK_ERROR,
                original_error=e,
            )
        except (KeyError, ValueError, TypeError) as e:
            # Handle response parsing errors
            raise UploadError(
                message=f"Invalid response format: {str(e)}",
                error_type=UploadErrorType.PARSE_ERROR,
                original_error=e,
            )
        except UploadError:
            # Re-throw exceptions that are already of type UploadError
            raise
        except Exception as e:
            # Handle other unexpected errors
            raise UploadError(
                message=f"Upload failed: {str(e)}",
                error_type=UploadErrorType.UNKNOWN,
                original_error=e,
            )


class AliyunOSSUploader(ImageUploader):
    """Alibaba Cloud OSS Image Uploader"""

    def __init__(
        self,
        access_key: str,
        access_key_secret: str,
        bucket_name: str,
        endpoint: str,
        region: str,
        use_internal: bool = False,
    ):
        """
        Initialize the Alibaba Cloud OSS uploader.

        Args:
            access_key: OSS access key ID
            access_key_secret: OSS access key secret
            bucket_name: OSS bucket name
            endpoint: OSS endpoint address
            region: OSS region
            use_internal: Whether to use the internal endpoint
        """
        self.access_key = access_key
        self.access_key_secret = access_key_secret
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.region = region
        self.use_internal = use_internal
        self.logger = get_image_create_logger()

        # Build the request URL
        if not endpoint.startswith(("http://", "https://")):
            self.base_url = f"https://{bucket_name}.{endpoint}"
        else:
            self.base_url = f"{endpoint}/{bucket_name}"

        self.logger.info(
            f"Initialized AliyunOSSUploader for bucket: {bucket_name}, region: {region}"
        )

    def _sign_request(
        self, method: str, path: str, headers: dict, content: bytes = b""
    ) -> dict:
        """
        Generate a signature for the OSS request.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            content: Request content

        Returns:
            Request headers containing the signature
        """
        # Calculate Content-MD5
        content_md5 = (
            base64.b64encode(hashlib.md5(content).digest()).decode("utf-8")
            if content
            else ""
        )

        # Set the date
        date = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

        # Update headers
        headers["Date"] = date
        if content_md5:
            headers["Content-MD5"] = content_md5
        headers["Content-Type"] = headers.get("Content-Type", "image/png")

        # Build CanonicalizedOSSHeaders
        oss_headers = []
        for key, value in sorted(headers.items()):
            if key.lower().startswith("x-oss-"):
                oss_headers.append(f"{key.lower()}:{value}")
        canonicalized_oss_headers = "\n".join(oss_headers)
        if canonicalized_oss_headers:
            canonicalized_oss_headers += "\n"

        # Build CanonicalizedResource
        canonicalized_resource = f"/{self.bucket_name}{path}"

        # Build StringToSign
        string_to_sign = f"{method}\n{content_md5}\n{headers.get('Content-Type', '')}\n{date}\n{canonicalized_oss_headers}{canonicalized_resource}"

        # Calculate the signature
        signature = base64.b64encode(
            hmac.new(
                self.access_key_secret.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("utf-8")

        # Add the Authorization header
        headers["Authorization"] = f"OSS {self.access_key}:{signature}"

        return headers

    def upload(self, file: bytes, filename: str) -> UploadResponse:
        """
        Upload an image to Alibaba Cloud OSS.

        Args:
            file: Image file binary data
            filename: File name (will be used as the OSS object key)

        Returns:
            UploadResponse: Upload response object

        Raises:
            UploadError: Thrown when the upload fails
        """
        # Log the start of the upload
        self.logger.info(
            f"Starting OSS upload for file: {filename}, size: {len(file)} bytes"
        )

        try:
            # Build the object path
            object_key = f"/{filename}"

            # Prepare request headers
            headers = {
                "Content-Type": "image/png",
                "x-oss-object-acl": "public-read",  # Set to public read
            }

            # Sign the request
            signed_headers = self._sign_request("PUT", object_key, headers, file)

            # Build the full URL
            upload_url = f"{self.base_url}{object_key}"
            self.logger.debug(f"OSS upload URL: {upload_url}")

            # Send the request
            response = requests.put(upload_url, data=file, headers=signed_headers)

            # Check the response status
            if response.status_code != 200:
                error_msg = f"OSS upload failed with status {response.status_code}, response: {response.text}"
                self.logger.error(f"OSS upload failed for {filename}: {error_msg}")
                raise UploadError(
                    message=f"OSS upload failed with status {response.status_code}",
                    error_type=UploadErrorType.SERVER_ERROR,
                    status_code=response.status_code,
                    details={"response": response.text},
                )

            # Build the access URL
            if self.endpoint.startswith(("http://", "https://")):
                access_url = f"{self.endpoint}/{self.bucket_name}{object_key}"
            else:
                access_url = f"https://{self.bucket_name}.{self.endpoint}{object_key}"

            # Build image metadata
            image_metadata = ImageMetadata(
                width=0,  # OSS PUT does not return image dimensions
                height=0,
                filename=filename,
                size=len(file),
                url=access_url,
                delete_url=None,  # OSS requires a separate delete operation
            )

            # Log the successful upload
            self.logger.info(f"OSS upload successful for {filename}, URL: {access_url}")

            return UploadResponse(
                success=True,
                code="success",
                message="Upload to Aliyun OSS success",
                data=image_metadata,
            )

        except requests.RequestException as e:
            error_msg = f"OSS upload request failed: {str(e)}"
            self.logger.error(
                f"OSS upload request failed for {filename}: {error_msg}", exc_info=True
            )
            raise UploadError(
                message=error_msg,
                error_type=UploadErrorType.NETWORK_ERROR,
                original_error=e,
            )
        except UploadError:
            # UploadError has already been logged, re-throw it
            raise
        except Exception as e:
            error_msg = f"OSS upload failed: {str(e)}"
            self.logger.error(
                f"OSS upload unexpected error for {filename}: {error_msg}",
                exc_info=True,
            )
            raise UploadError(
                message=error_msg, error_type=UploadErrorType.UNKNOWN, original_error=e
            )


class CloudFlareImgBedUploader(ImageUploader):
    """CloudFlare Image Bed Uploader"""

    def __init__(self, auth_code: str, api_url: str, upload_folder: str = ""):
        """
        Initialize the CloudFlare Image Bed uploader.

        Args:
            auth_code: Authentication code
            api_url: Upload API address
            upload_folder: Upload folder path (optional)
        """
        self.auth_code = auth_code
        self.api_url = api_url
        self.upload_folder = upload_folder

    def upload(self, file: bytes, filename: str) -> UploadResponse:
        """
        Upload an image to the CloudFlare Image Bed.

        Args:
            file: Image file binary data
            filename: File name

        Returns:
            UploadResponse: Upload response object

        Raises:
            UploadError: Thrown when the upload fails
        """
        try:
            # Prepare request URL parameters
            params = []
            if self.upload_folder:
                params.append(f"uploadFolder={self.upload_folder}")
            if self.auth_code:
                params.append(f"authCode={self.auth_code}")
            params.append("uploadNameType=origin")

            request_url = f"{self.api_url}?{'&'.join(params)}"

            # Prepare file data
            files = {"file": (filename, file)}

            # Send the request
            response = requests.post(request_url, files=files)

            # Check the response status
            response.raise_for_status()

            # Parse the response
            result = response.json()

            # Validate the response format
            if not result or not isinstance(result, list) or len(result) == 0:
                raise UploadError(
                    message="Invalid response format",
                    error_type=UploadErrorType.PARSE_ERROR,
                )

            # Get the file URL
            file_path = result[0].get("src")
            if not file_path:
                raise UploadError(
                    message="Missing file URL in response",
                    error_type=UploadErrorType.PARSE_ERROR,
                )

            # Build the full URL (if a relative path is returned)
            base_url = self.api_url.split("/upload")[0]
            full_url = (
                file_path
                if file_path.startswith(("http://", "https://"))
                else f"{base_url}{file_path}"
            )

            # Build image metadata (note: CloudFlare-ImgBed does not return all metadata, so some fields have default values)
            image_metadata = ImageMetadata(
                width=0,  # CloudFlare-ImgBed does not return width
                height=0,  # CloudFlare-ImgBed does not return height
                filename=filename,
                size=0,  # CloudFlare-ImgBed does not return size
                url=full_url,
                delete_url=None,  # CloudFlare-ImgBed does not return a delete URL
            )

            return UploadResponse(
                success=True,
                code="success",
                message="Upload success",
                data=image_metadata,
            )

        except requests.RequestException as e:
            # Handle network request related errors
            raise UploadError(
                message=f"Upload request failed: {str(e)}",
                error_type=UploadErrorType.NETWORK_ERROR,
                original_error=e,
            )
        except (KeyError, ValueError, TypeError, IndexError) as e:
            # Handle response parsing errors
            raise UploadError(
                message=f"Invalid response format: {str(e)}",
                error_type=UploadErrorType.PARSE_ERROR,
                original_error=e,
            )
        except UploadError:
            # Re-throw exceptions that are already of type UploadError
            raise
        except Exception as e:
            # Handle other unexpected errors
            raise UploadError(
                message=f"Upload failed: {str(e)}",
                error_type=UploadErrorType.UNKNOWN,
                original_error=e,
            )


class ImageUploaderFactory:
    @staticmethod
    def create(provider: str, **credentials) -> ImageUploader:
        if provider == "smms":
            return SmMsUploader(credentials["api_key"])
        elif provider == "qiniu":
            return QiniuUploader(credentials["access_key"], credentials["secret_key"])
        elif provider == "picgo":
            api_url = credentials.get("api_url") or "https://www.picgo.net/api/1/upload"
            return PicGoUploader(credentials["api_key"], api_url)
        elif provider == "cloudflare_imgbed":
            return CloudFlareImgBedUploader(
                credentials["auth_code"],
                credentials["base_url"],
                credentials.get("upload_folder", ""),
            )
        elif provider == "aliyun_oss":
            return AliyunOSSUploader(
                credentials["access_key"],
                credentials["access_key_secret"],
                credentials["bucket_name"],
                credentials["endpoint"],
                credentials["region"],
                credentials.get("use_internal", False),
            )
        raise ValueError(f"Unknown provider: {provider}")
