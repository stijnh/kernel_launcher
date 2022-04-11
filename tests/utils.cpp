#include "catch.hpp"
#include "kernel_launcher/value.hpp"

namespace kl = kernel_launcher;

TEST_CASE("Type") {
    using kl::Type;
    using kl::type_name;
    using kl::type_of;

    Type int_type = Type::of<int>();
    Type float_type = Type::of<float>();
    CHECK(int_type == int_type);
    CHECK(int_type != float_type);
    CHECK(int_type.name() == "int");
    CHECK(float_type.name() == "float");
    CHECK(type_of(123) == int_type);
    CHECK(type_of(123.0f) == float_type);
    CHECK(type_of(123.0) != float_type);
    CHECK(type_name(123) == "int");
    CHECK(type_name(123.0f) == "float");
    CHECK(type_name(123.0) != "float");
}

TEST_CASE("TemplateArg") {
    using kl::template_arg;
    using kl::TemplateArg;
    using kl::type_of;

    CHECK(template_arg(true).get() == "(bool)true");
    CHECK(template_arg(false).get() == "(bool)false");
    CHECK(template_arg((char)1).get() == "(char)1");
    CHECK(template_arg((unsigned char)1).get() == "(unsigned char)1");
    CHECK(template_arg((signed char)1).get() == "(signed char)1");
    CHECK(template_arg(1).get() == "(int)1");
    CHECK(template_arg(1u).get() == "(unsigned int)1");
    CHECK(template_arg(1l).get() == "(long)1");
    CHECK(template_arg(1.0).get() == "(double)1.000000");
    CHECK(template_arg(1.0f).get() == "(float)1.000000");
    CHECK(template_arg(1.0l).get() == "(long double)1.000000");
    CHECK(TemplateArg(type_of(1)).get() == "int");
    CHECK(TemplateArg::from_string("abc").get() == "abc");
}