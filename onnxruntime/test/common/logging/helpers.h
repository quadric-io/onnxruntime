// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"

// gmock prints the arguments of mocked calls, and would print a bare Timestamp with C++20 std::chrono's operator<<,
// which needs macOS 13.3+. The mocks receive the Timestamp wrapped in this type, which prints through timestamp_ns.
struct MockTimestamp {
  ::onnxruntime::logging::Timestamp value;
};

inline std::ostream& operator<<(std::ostream& os, const MockTimestamp& timestamp) {
  // handle ambiguity with C++20 where date and std::chrono have operator<<
  return ::onnxruntime::logging::timestamp_ns::operator<<(os, timestamp.value);
}

class MockSink : public ::onnxruntime::logging::ISink {
 public:
  using ISink::ISink;

  MOCK_METHOD3(SendImpl, void(MockTimestamp timestamp,
                              const std::string& logger_id,
                              const ::onnxruntime::logging::Capture& message));

 private:
  void SendImpl(const ::onnxruntime::logging::Timestamp& timestamp,
                const std::string& logger_id,
                const ::onnxruntime::logging::Capture& message) override {
    SendImpl(MockTimestamp{timestamp}, logger_id, message);
  }
};

class MockEtwSink : public MockSink {
 public:
  MockEtwSink() : MockSink(onnxruntime::logging::SinkType::EtwSink) {}
};

// The ACTION*() macros trigger warning C4100 (unreferenced formal
// parameter) in MSVC with -W4.  Unfortunately they cannot be fixed in
// the macro definition, as the warnings are generated when the macro
// is expanded and macro expansion cannot contain #pragma.  Therefore
// we suppress them here.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

ACTION(PrintArgs) {
  // const MockTimestamp &timestamp, const std::string &logger_id, const Message &message
  //                      arg0                          arg1                        arg2
  std::cout << arg1 << "@" << arg0 << " "
            << arg2.SeverityPrefix() << ":" << arg2.Category() << ":"
            << arg2.Location().ToString(::onnxruntime::CodeLocation::kFilenameAndPath) << " " << arg2.Message() << std::endl;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif
