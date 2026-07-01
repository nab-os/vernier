/* 
 * This file is part of the VERNIER Library.
 *
 * Copyright (c) 2018-2025 CNRS, ENSMM, UMLP.
 */

#ifndef LAYOUT_HPP
#define LAYOUT_HPP

#include "PatternLayout.hpp"
#include "PeriodicPatternLayout.hpp"
#include "BitmapPatternLayout.hpp"
#include "FingerprintPatternLayout.hpp"
#include "MegarenaPatternLayout.hpp"
#include "HPCodePatternLayout.hpp"
#include "CustomPatternLayout.hpp"
#include <memory>

namespace vernier {

    class Layout {
    public:

        static std::unique_ptr<PatternLayout> newInstance(const std::string& classname) {
            std::unique_ptr<PatternLayout> layout;
            if (classname == "PeriodicPattern") {
                layout.reset(new PeriodicPatternLayout());
            } else if (classname == "BitmapPattern") {
                layout.reset(new BitmapPatternLayout());
            } else if (classname == "FingerprintPattern") {
                layout.reset(new FingerprintPatternLayout());
            } else if (classname == "MegarenaPattern") {
                layout.reset(new MegarenaPatternLayout());
            } else if (classname == "HPCodePattern") {
                layout.reset(new HPCodePatternLayout());
            } else if (classname == "CustomPattern") {
                layout.reset(new CustomPatternLayout());
            } else {
                throw Exception(classname + " is not a valid class name for a pattern layout.");
            }
            return layout;
        }

        static std::unique_ptr<PatternLayout> loadFromJSON(const std::string& filename) {
            BufferedReader bufferedReader(filename);
            rapidjson::Document document;
            document.ParseInsitu(bufferedReader.data());
            if (!document.IsObject()) {
                throw Exception(filename + " is not a valid JSON file.");
            }
            if (document.MemberBegin() == document.MemberEnd()) {
                throw Exception(filename + " is empty.");
            }
            std::string classname = document.MemberBegin()->name.GetString();
            std::unique_ptr<PatternLayout> layout = newInstance(classname);
            layout->readJSON(document.MemberBegin()->value);
            return layout;
        }
        
    };

}

#endif