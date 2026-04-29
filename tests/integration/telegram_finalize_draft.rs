use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use zeroclaw::channels::Channel;
use zeroclaw::channels::telegram::TelegramChannel;

fn test_channel(mock_url: &str) -> TelegramChannel {
    TelegramChannel::new("TEST_TOKEN".into(), vec!["*".into()], false)
        .with_api_base(mock_url.to_string())
}

fn telegram_ok_response(message_id: i64) -> serde_json::Value {
    json!({
        "ok": true,
        "result": {
            "message_id": message_id,
            "chat": {"id": 123},
            "text": "ok"
        }
    })
}

fn telegram_error_response(description: &str) -> serde_json::Value {
    json!({
        "ok": false,
        "error_code": 400,
        "description": description,
    })
}

#[tokio::test]
async fn finalize_draft_treats_not_modified_as_success() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/editMessageText"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(telegram_error_response(
                "Bad Request: message is not modified",
            )),
        )
        .mount(&server)
        .await;

    let channel = test_channel(&server.uri());
    let result = channel.finalize_draft("123", "42", "final text").await;

    assert!(
        result.is_ok(),
        "not modified should be treated as success, got: {result:?}"
    );

    let requests = server
        .received_requests()
        .await
        .expect("requests should be captured");
    assert_eq!(requests.len(), 1, "should stop after first edit response");
    assert_eq!(requests[0].url.path(), "/botTEST_TOKEN/editMessageText");
}

#[tokio::test]
async fn finalize_draft_plain_retry_treats_not_modified_as_success() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/editMessageText"))
        .and(body_partial_json(json!({
            "chat_id": "123",
            "message_id": 42,
            "parse_mode": "HTML",
        })))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(telegram_error_response("Bad Request: can't parse entities")),
        )
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/editMessageText"))
        .and(body_partial_json(json!({
            "chat_id": "123",
            "message_id": 42,
            "text": "Use **bold**",
        })))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(telegram_error_response(
                "Bad Request: message is not modified",
            )),
        )
        .expect(1)
        .mount(&server)
        .await;

    let channel = test_channel(&server.uri());
    let result = channel.finalize_draft("123", "42", "Use **bold**").await;

    assert!(
        result.is_ok(),
        "plain retry should accept not modified, got: {result:?}"
    );

    let requests = server
        .received_requests()
        .await
        .expect("requests should be captured");
    assert_eq!(requests.len(), 2, "should only attempt the two edit calls");
}

#[tokio::test]
async fn finalize_draft_skips_send_message_when_delete_fails() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/editMessageText"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(telegram_error_response(
                "Bad Request: message cannot be edited",
            )),
        )
        .expect(2)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/deleteMessage"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(telegram_error_response(
                "Bad Request: message to delete not found",
            )),
        )
        .expect(1)
        .mount(&server)
        .await;

    let channel = test_channel(&server.uri());
    let result = channel.finalize_draft("123", "42", "final text").await;

    assert!(
        result.is_ok(),
        "delete failure should skip sendMessage instead of erroring, got: {result:?}"
    );

    let requests = server
        .received_requests()
        .await
        .expect("requests should be captured");
    assert_eq!(
        requests
            .iter()
            .filter(|req| req.url.path() == "/botTEST_TOKEN/sendMessage")
            .count(),
        0,
        "sendMessage should be skipped when deleteMessage fails"
    );
}

#[tokio::test]
async fn finalize_draft_sends_fresh_message_after_successful_delete() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/editMessageText"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(telegram_error_response(
                "Bad Request: message cannot be edited",
            )),
        )
        .expect(2)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/deleteMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(42)))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/sendMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(43)))
        .expect(1)
        .mount(&server)
        .await;

    let channel = test_channel(&server.uri());
    let result = channel.finalize_draft("123", "42", "final text").await;

    assert!(
        result.is_ok(),
        "successful delete should allow safe sendMessage fallback, got: {result:?}"
    );

    let requests = server
        .received_requests()
        .await
        .expect("requests should be captured");
    assert_eq!(
        requests
            .iter()
            .filter(|req| req.url.path() == "/botTEST_TOKEN/sendMessage")
            .count(),
        1,
        "sendMessage should be attempted exactly once after delete succeeds"
    );
}

/// Test that finalize_draft falls back to sendMessage after delete when edit fails.
#[tokio::test]
async fn finalize_draft_delete_resend_routes_correctly() {
    let server = MockServer::start().await;

    // Both edit attempts fail → delete → send fallback
    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/editMessageText"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(telegram_error_response(
                "Bad Request: message cannot be edited",
            )),
        )
        .expect(2)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/deleteMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(42)))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/sendMessage"))
        .and(body_partial_json(json!({
            "chat_id": "123",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(43)))
        .expect(1)
        .mount(&server)
        .await;

    let channel = test_channel(&server.uri());
    let result = channel.finalize_draft("123", "42", "final text").await;

    assert!(
        result.is_ok(),
        "finalize_draft should succeed after delete fallback, got: {result:?}"
    );

    let requests = server
        .received_requests()
        .await
        .expect("requests should be captured");

    let send_req = requests
        .iter()
        .find(|req| req.url.path() == "/botTEST_TOKEN/sendMessage")
        .expect("sendMessage should have been called");

    let body: serde_json::Value = send_req
        .body_json()
        .expect("sendMessage body should be JSON");

    assert_eq!(
        body.get("chat_id").and_then(|v| v.as_str()),
        Some("123"),
        "sendMessage should be called with correct chat_id"
    );
}

/// Test that finalize_draft with oversized text falls back to chunked send.
#[tokio::test]
async fn finalize_draft_oversized_fallback_routes_correctly() {
    let server = MockServer::start().await;

    // Oversized text + invalid draft message_id → chunked send fallback.
    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/sendMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(44)))
        .mount(&server)
        .await;

    let channel = test_channel(&server.uri());
    // 4200 "a" chars → chunked into 2 sends (first: 4066, second: 134)
    let long_text = "a".repeat(4200);

    let result = channel
        .finalize_draft("123", "not-a-number", &long_text)
        .await;

    assert!(
        result.is_ok(),
        "finalize_draft with oversized text should succeed, got: {result:?}"
    );

    let requests = server
        .received_requests()
        .await
        .expect("requests should be captured");

    let send_reqs: Vec<_> = requests
        .iter()
        .filter(|req| req.url.path() == "/botTEST_TOKEN/sendMessage")
        .collect();

    assert!(
        !send_reqs.is_empty(),
        "sendMessage should have been called at least once"
    );

    // Verify the FIRST sendMessage has correct chat_id
    let body: serde_json::Value = send_reqs[0]
        .body_json()
        .expect("sendMessage body should be JSON");

    assert_eq!(
        body.get("chat_id").and_then(|v| v.as_str()),
        Some("123"),
        "sendMessage should be called with correct chat_id"
    );
}

// ── Attachment + reply_parameters tests ────────────────────────────────────────

/// Test that finalize_draft with an image attachment routes to sendPhoto.
#[tokio::test]
async fn finalize_draft_attachment_sends_photo() {
    let server = MockServer::start().await;

    // deleteMessage succeeds
    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/deleteMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(42)))
        .expect(1)
        .mount(&server)
        .await;

    // sendMessage (text portion) succeeds
    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/sendMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(43)))
        .expect(1)
        .mount(&server)
        .await;

    // sendPhoto is called
    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/sendPhoto"))
        .and(body_partial_json(json!({
            "chat_id": "123",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ok": true,
            "result": {
                "message_id": 44,
                "chat": {"id": 123},
                "photo": {"file_id": "abc"}
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let channel = test_channel(&server.uri());
    let result = channel
        .finalize_draft(
            "123",
            "42",
            "Look at this [IMAGE:https://example.com/img.jpg]",
        )
        .await;

    assert!(
        result.is_ok(),
        "finalize_draft with attachment should succeed, got: {result:?}"
    );

    let requests = server.received_requests().await.expect("requests captured");
    let photo_req = requests
        .iter()
        .find(|r| r.url.path() == "/botTEST_TOKEN/sendPhoto")
        .expect("sendPhoto should have been called");

    let body: serde_json::Value = photo_req.body_json().expect("body should be JSON");
    assert_eq!(
        body.get("chat_id").and_then(|v| v.as_str()),
        Some("123"),
        "sendPhoto should be called with correct chat_id"
    );
}

/// Test that finalize_draft with a document attachment routes to sendDocument.
#[tokio::test]
async fn finalize_draft_attachment_sends_document() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/deleteMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(42)))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/sendMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(telegram_ok_response(43)))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/botTEST_TOKEN/sendDocument"))
        .and(body_partial_json(json!({
            "chat_id": "123",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "ok": true,
            "result": {
                "message_id": 45,
                "chat": {"id": 123},
                "document": {"file_id": "doc123"}
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let channel = test_channel(&server.uri());
    let result = channel
        .finalize_draft(
            "123",
            "42",
            "Here is the file [DOCUMENT:https://example.com/file.pdf]",
        )
        .await;

    assert!(
        result.is_ok(),
        "finalize_draft with document attachment should succeed, got: {result:?}"
    );

    let requests = server.received_requests().await.expect("requests captured");
    let doc_req = requests
        .iter()
        .find(|r| r.url.path() == "/botTEST_TOKEN/sendDocument")
        .expect("sendDocument should have been called");

    let body: serde_json::Value = doc_req.body_json().expect("body should be JSON");
    assert_eq!(
        body.get("chat_id").and_then(|v| v.as_str()),
        Some("123"),
        "sendDocument should be called with correct chat_id"
    );
}
