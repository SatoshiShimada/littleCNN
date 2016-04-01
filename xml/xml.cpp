
#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/forearch.hpp>
#include <boost/lexical_cast.hpp>

int main(int argc, char **argv)
{
	boost::property_tree::ptree pt;
	read_xml("net.xml", pt);

	if(boost::optional<std::string> str = pt.get_optional<std::string>("littleCNN.network.name")) {
		std::cout << str.get() << std::endl;
} else {
		std::cout << "root.str is nothing" << std::endl;
	}

	BOOST_FOREACH(const boost::property_tree::ptree::value_type& child, pt.get_child("littleCNN.network.layer")) {
		const int value = boost::lexical_cast<int>(child.second.data());
		std::cout << value << std::endl;
	}
	return 0;
}

