/*global $*/

$(document).ready(function() {
  $("#prediction-form").on("submit", function() {
      $("#predict-button").prop("disabled", true); // 버튼 비활성화
      $("#loading-message").show(); // 메시지 표시
  });
});