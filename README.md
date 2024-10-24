## Bước 1: Tạo Môi Trường Ảo

Trong thư mục dự án của bạn, mở terminal (hoặc command prompt) và chạy lệnh sau để tạo môi trường ảo:

```bash
python -m venv venv
```

`venv` ở sau là tên thư mục của môi trường ảo mà bạn sẽ tạo ra. Bạn có thể đặt tên khác nếu muốn.

## Bước 2: Kích Hoạt Môi Trường Ảo

Sau khi môi trường ảo được tạo, bạn cần kích hoạt nó:

  ```bash
  myenv\Scripts\activate
  ```

Khi kích hoạt thành công, bạn sẽ thấy tên của môi trường ảo xuất hiện trước dấu nhắc lệnh trong terminal, ví dụ: `(myenv)`.

## Bước 3: Cài Đặt Các Gói Thư Viện

Khi môi trường ảo đã được kích hoạt, bạn có thể cài đặt các thư viện cần thiết bằng lệnh `pip`. Ví dụ:

```bash
pip install package_name
```

Để lưu lại danh sách các gói đã cài đặt, bạn có thể tạo file `requirements.txt` bằng lệnh:

```bash
pip freeze > requirements.txt
```

## Bước 4: Cài Đặt Thư Viện Từ `requirements.txt`

Khi bạn chia sẻ dự án với người khác, họ có thể cài đặt các thư viện cần thiết bằng lệnh sau:

```bash
pip install -r requirements.txt
```

## Bước 5: Tắt Môi Trường Ảo

Khi hoàn thành công việc, bạn có thể tắt môi trường ảo bằng cách chạy lệnh sau:

```bash
deactivate
```