
#include <cstdio>
#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

int main(int argc, char **argv)
{
	char name[100];

	boost::property_tree::ptree pt;
	read_xml("logic.xml", pt);

	if(boost::optional<std::string> str = pt.get_optional<std::string>("littleCNN.network.name")) {
		std::cout << str.get() << std::endl;
} else {
		std::cout << "root.str is nothing" << std::endl;
	}

	for(int i = 0; ; i++) {
		try {
			sprintf(name, "littleCNN.network.layer%d", i + 1);
			printf("%s\n", name);
			BOOST_FOREACH(const boost::property_tree::ptree::value_type& child, pt.get_child(name)) {
				const std::string value = boost::lexical_cast<std::string>(child.second.data());
				std::cout << value << std::endl;
			}
		} catch(...) {
			break;
		}
	}
	return 0;
}

