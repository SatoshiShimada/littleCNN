
#include <cstdio>
#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

int append_parameter(std::string first, std::string second)
{
	/* TODO: impliment to class (layer parameter) */
	if(first == "type") {
	} else if(first == "inputNum") {
	} else if(first == "outputNum") {
	} else if(first == "activation") {
	} else if(first == "learningRate") {
	}
	return 0;
}

int load_network_parameter_from_xml(std::string filename)
{
    char name[100];

    boost::property_tree::ptree pt;
    read_xml(filename, pt);

    if(boost::optional<std::string> str = pt.get_optional<std::string>("littleCNN.network.name")) {
        std::cout << str.get() << std::endl;
    } else {
        std::cout << "root.str is nothing" << std::endl;
    }

    /* decode xml */
    for(int i = 0; ; i++) {
        try {
            sprintf(name, "littleCNN.network.layer%d", i + 1);
            printf("%s\n", name);
            BOOST_FOREACH(const boost::property_tree::ptree::value_type& child, pt.get_child(name)) {
                const std::string first = boost::lexical_cast<std::string>(child.first.data());
                const std::string second = boost::lexical_cast<std::string>(child.second.data());
                std::cout << first << ": " << second << std::endl;
				append_parameter(first, second);
            }
        } catch(...) {
            break;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    load_network_parameter_from_xml("logic.xml");
	return 0;
}

