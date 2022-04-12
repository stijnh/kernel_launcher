#include "kernel_launcher/config.hpp"

#include "catch.hpp"

TEST_CASE("test Config") {
    using namespace kernel_launcher;

    Config config;
    TunableParam x(
        "x",
        type_of<int>(),
        {TunableValue(1), 2, 3},
        TunableValue(1));
    TunableParam y(
        "y",
        type_of<bool>(),
        {TunableValue(true), false},
        TunableValue(true));
    TunableParam z(
        "x",
        type_of<int>(),
        {TunableValue(1), 2, 3},
        TunableValue(1));

    CHECK(config.size() == 0);
    CHECK_THROWS(config[x]);

    config.insert(x, x[0]);
    config.insert(y, y[1]);

    CHECK(config.size() == 2);
    CHECK(config[x] == TunableValue(1));
    CHECK(config[y] == TunableValue(false));
    CHECK_THROWS(config[z]);

    Config copy_config = Config(config);

    Config other_config;
    other_config.insert(y, y[1]);
    other_config.insert(x, x[0]);

    Config another_config;
    another_config.insert(x, x[0]);
    another_config.insert(z, z[1]);

    CHECK(copy_config == config);
    CHECK(other_config == config);
    CHECK(another_config != config);
    CHECK(other_config != another_config);

    nlohmann::json obj = {
        {"x", 1},
        {"y", false},
    };
    CHECK(config.to_json() == obj);
}

/*
bool is_valid(const Config& config) const;
Config random_config() const;
Config default_config() const;
ConfigIterator iterate() const;
Config load_config(const nlohmann::json& obj) const;
nlohmann::json to_json() const;
*/

TEST_CASE("test ConfigSpace") {
    using namespace kernel_launcher;

    ConfigSpace space;
    auto foo = space.tune("foo", {1, 2, 3});  // initializer list
    auto bar = space.tune("bar", std::vector<long> {1, 2, 3});  // std::vector
    space.restrict(foo <= bar);

    std::string strings[3] = {"hi", "bye", "default"};
    auto baz = space.tune(
        "baz",
        strings,
        strings + 3,
        std::string("default"));  // iterator

    SECTION("basics") {
        CHECK_THROWS(space.tune("empty", std::vector<int> {}));

        CHECK(space["foo"].name() == "foo");
        CHECK(space["foo"].type() == Type::of<int>());
        CHECK(space["foo"].default_value() == 1);
        CHECK(space["foo"].size() == 3);
        CHECK(space["foo"][1] == 2);
        CHECK_THROWS(space["foo"][3]);

        CHECK(space["foo"] == space["foo"]);
        CHECK(space["foo"] != space["bar"]);
        CHECK_THROWS(space["empty"]);

        CHECK(space.size() == 27);
    }

    SECTION("default config") {
        Config def = space.default_config();
        CHECK(space.is_valid(def));
        CHECK(def[foo] == 1);
        CHECK(def[bar] == 1);
        CHECK(def[baz] == "default");
    }

    SECTION("random config") {
        Config rand = space.random_config();
        CHECK(space.is_valid(rand));
    }

    SECTION("make config") {
        Config config;
        config.insert(foo.parameter(), 1);
        config.insert(bar.parameter(), 2);
        config.insert(baz.parameter(), "hi");
        CHECK(space.is_valid(config));

        config = Config();
        CHECK(space.is_valid(config) == false);

        config = Config();
        config.insert(foo.parameter(), 2);
        config.insert(bar.parameter(), 1);
        config.insert(baz.parameter(), "hi");
        CHECK(space.is_valid(config) == false);

        config = Config();
        config.insert(foo.parameter(), 1);
        config.insert(bar.parameter(), 2);
        config.insert(baz.parameter(), "invalid valid");
        CHECK(space.is_valid(config) == false);
    }

    SECTION("get valid config") {
        Config valid;
        CHECK(space.get(6, valid) == true);
        CHECK(space.is_valid(valid) == true);
        CHECK(valid[foo] == 1);
        CHECK(valid[bar] == 3);
        CHECK(valid[baz] == "hi");
    }

    SECTION("get invalid config") {
        Config invalid;
        CHECK(space.get(1, invalid) == false);
        CHECK(space.is_valid(invalid) == false);
        CHECK(invalid[foo] == 2);
        CHECK(invalid[bar] == 1);
        CHECK(invalid[baz] == "hi");
    }

    SECTION("iterator") {
        Config config;
        std::vector<Config> visited;
        ConfigIterator it = space.iterate();
        while (it.next(config)) {
            for (const auto& c : visited) {
                CHECK(c != config);
            }

            visited.push_back(config);
        }

        CHECK(visited.size() == 18);
    }

    SECTION("load json") {
        Config config;
        config.insert(foo.parameter(), 1);
        config.insert(bar.parameter(), 2);
        config.insert(baz.parameter(), "hi");

        nlohmann::json obj = {
            {"foo", 1},
            {"bar", 2},
            {"baz", "hi"},
        };

        CHECK(config.to_json() == obj);
        CHECK(space.load_config(obj) == config);
    }
}
