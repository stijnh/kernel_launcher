#include "kernel_launcher/value.hpp"

#include "catch.hpp"

TEST_CASE("test TunableParam") {
    using kernel_launcher::TunableParam;
    using kernel_launcher::TunableValue;
    using kernel_launcher::Type;

    std::vector<TunableValue> values = {1, 2, 3};

    TunableParam param("foo", Type::of<int>(), values, 1);
    TunableParam param2("foo", Type::of<int>(), values, 1);

    CHECK(param.name() == "foo");
    CHECK((param.type() == Type::of<int>()));
    CHECK(param.default_value() == 1);
    CHECK(param.size() == 3);
    CHECK(param[1] == 2);
    CHECK_THROWS(param[100]);
    CHECK(param == param);
    CHECK(param != param2);
}

TEST_CASE("test TunableValue") {
    using kernel_launcher::TunableValue;

    SECTION("empty") {
        TunableValue val;
        CHECK(val.is_empty());
        CHECK_FALSE(val.is_uint());
        CHECK_FALSE(val.is_int());
        CHECK_FALSE(val.is_long());
        CHECK_FALSE(val.is_string());
        CHECK_FALSE(val.is_bool());
        CHECK_FALSE(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK_FALSE(val.is_string());

        CHECK(val.to_string() == "");
        CHECK(val.to_json() == nullptr);
    }

    SECTION("int") {
        TunableValue val(0);
        CHECK_FALSE(val.is_empty());
        CHECK(val.is_uint());
        CHECK(val.is_int());
        CHECK(val.is_long());
        CHECK(val.is_string());
        CHECK(val.is_bool());
        CHECK(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_bool() == false);
        CHECK(val.to_int() == 0);
        CHECK(val == TunableValue(0));
        CHECK(val != TunableValue(123.0));
        CHECK(val.to_string() == "0");
        CHECK(val.to_json() == 0);
        CHECK_THROWS(val.to_double());
    }

    SECTION("int") {
        TunableValue val(1);
        CHECK_FALSE(val.is_empty());
        CHECK(val.is_uint());
        CHECK(val.is_int());
        CHECK(val.is_long());
        CHECK(val.is_string());
        CHECK(val.is_bool());
        CHECK(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_bool() == true);
        CHECK(val.to_int() == 1);
        CHECK(val == TunableValue(1));
        CHECK(val != TunableValue(0.0));
        CHECK(val.to_string() == "1");
        CHECK(val.to_json() == 1);
        CHECK_THROWS(val.to_double());
    }

    SECTION("int") {
        TunableValue val(-1);
        CHECK_FALSE(val.is_empty());
        CHECK_FALSE(val.is_uint());
        CHECK(val.is_int());
        CHECK(val.is_long());
        CHECK(val.is_string());
        CHECK_FALSE(val.is_bool());
        CHECK(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_int() == -1);
        CHECK(val == TunableValue(-1));
        CHECK(val != TunableValue(-1.0));
        CHECK(val.to_string() == "-1");
        CHECK(val.to_json() == -1);
        CHECK_THROWS(val.to_double());
    }

    SECTION("int") {
        TunableValue val(300);
        CHECK_FALSE(val.is_empty());
        CHECK(val.is_uint());
        CHECK(val.is_int());
        CHECK(val.is_long());
        CHECK(val.is_string());
        CHECK_FALSE(val.is_bool());
        CHECK_FALSE(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_int() == 300);
        CHECK(val == TunableValue(300));
        CHECK(val != TunableValue(300.0));
        CHECK(val.to_string() == "300");
        CHECK(val.to_json() == 300);
        CHECK_THROWS(val.to_double());
    }

    SECTION("bool true") {
        TunableValue val(true);
        CHECK_FALSE(val.is_empty());
        CHECK(val.is_uint());
        CHECK(val.is_int());
        CHECK(val.is_long());
        CHECK(val.is_string());
        CHECK(val.is_bool());
        CHECK(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_int() == 1);
        CHECK(val.to_bool() == true);
        CHECK(val == TunableValue(true));
        CHECK(val != TunableValue(false));
        CHECK(val != TunableValue(300.0));
        CHECK(val.to_string() == "true");
        CHECK(val.to_json() == true);
        CHECK_THROWS(val.to_double());
    }

    SECTION("bool false") {
        TunableValue val(false);
        CHECK_FALSE(val.is_empty());
        CHECK(val.is_uint());
        CHECK(val.is_int());
        CHECK(val.is_long());
        CHECK(val.is_string());
        CHECK(val.is_bool());
        CHECK(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_int() == 0);
        CHECK(val.to_bool() == false);
        CHECK(val == TunableValue(false));
        CHECK(val == TunableValue(false));
        CHECK(val != TunableValue(300.0));
        CHECK(val.to_string() == "false");
        CHECK(val.to_json() == false);
        CHECK_THROWS(val.to_double());
    }

    SECTION("double") {
        TunableValue val(123.0);
        CHECK_FALSE(val.is_empty());
        CHECK_FALSE(val.is_uint());
        CHECK_FALSE(val.is_int());
        CHECK_FALSE(val.is_long());
        CHECK(val.is_string());
        CHECK_FALSE(val.is_bool());
        CHECK_FALSE(val.is_char());
        CHECK(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_double() == 123);
        CHECK(val == TunableValue(123.0));
        CHECK(val != TunableValue(123));
        CHECK(val.to_string() == "123.000000");
        CHECK(val.to_json() == 123.0);
        CHECK_THROWS(val.to_int());
    }

    SECTION("string empty") {
        TunableValue val("");

        CHECK_FALSE(val.is_empty());
        CHECK_FALSE(val.is_uint());
        CHECK_FALSE(val.is_int());
        CHECK_FALSE(val.is_long());
        CHECK(val.is_string());
        CHECK_FALSE(val.is_bool());
        CHECK_FALSE(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_string() == "");
        CHECK(val == TunableValue(""));
        CHECK(val != TunableValue());
        CHECK(val.to_string() == "");
        CHECK(val.to_json() == "");
        CHECK_THROWS(val.to_bool());
    }

    SECTION("string int") {
        TunableValue val("123");

        CHECK_FALSE(val.is_empty());
        CHECK_FALSE(val.is_uint());
        CHECK_FALSE(val.is_int());
        CHECK_FALSE(val.is_long());
        CHECK(val.is_string());
        CHECK_FALSE(val.is_bool());
        CHECK_FALSE(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_string() == "123");
        CHECK(val == TunableValue("123"));
        CHECK(val != TunableValue(123));
        CHECK(val.to_string() == "123");
        CHECK(val.to_json() == "123");
        CHECK_THROWS(val.to_int());
    }

    SECTION("string char") {
        TunableValue val("f");

        CHECK_FALSE(val.is_empty());
        CHECK_FALSE(val.is_uint());
        CHECK_FALSE(val.is_int());
        CHECK_FALSE(val.is_long());
        CHECK(val.is_string());
        CHECK_FALSE(val.is_bool());
        CHECK_FALSE(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_string() == "f");
        CHECK(val == TunableValue("f"));
        CHECK(val != TunableValue('f'));
        CHECK(val.to_string() == "f");
        CHECK(val.to_json() == "f");
        CHECK_THROWS(val.to_int());
    }

    SECTION("string looooong") {
        static constexpr const char* MSG =
            "this is a long string that will defeat SSO";
        TunableValue val(MSG);

        CHECK_FALSE(val.is_empty());
        CHECK_FALSE(val.is_uint());
        CHECK_FALSE(val.is_int());
        CHECK_FALSE(val.is_long());
        CHECK(val.is_string());
        CHECK_FALSE(val.is_bool());
        CHECK_FALSE(val.is_char());
        CHECK_FALSE(val.is_double());
        CHECK(val.is_string());

        CHECK(val.to_string() == MSG);
        CHECK(val == TunableValue(MSG));
        CHECK(val != TunableValue());
        CHECK(val.to_string() == MSG);
        CHECK(val.to_json() == MSG);
        CHECK_THROWS(val.to_int());
    }
}
