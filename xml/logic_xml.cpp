
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

class LayerParameters
{
public:
	 int append_parameter(std::string first, std::string second)
	 {
		  if(first == "type") {
			   type = second;
		  } else if(first == "inputNum") {
			   inputNum = second;
		  } else if(first == "outputNum") {
			   outputNum = second;
		  } else if(first == "activation") {
			   activation = second;
		  } else if(first == "learningRate") {
			   lr = second;
		  }
		  return 0;
	 }

	 int print_parameters(void)
	 {
		  std::cout << "type: " << type << std::endl;
		  std::cout << "input num: " << inputNum << std::endl;
		  std::cout << "output num: " << outputNum << std::endl;
		  std::cout << "activation function: " << activation << std::endl;
		  std::cout << "learning rate: " << lr << std::endl;
		  return 0;
	 }
	
private:
	std::string type;
	std::string inputNum;
	std::string outputNum;
	std::string activation;
	std::string lr;
};

int load_network_parameter_from_xml(std::string filename)
{
    char name[100];

	LayerParameters param;
	std::vector<LayerParameters> params;
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
			LayerParameters param;
            BOOST_FOREACH(const boost::property_tree::ptree::value_type& child, pt.get_child(name)) {
                const std::string first = boost::lexical_cast<std::string>(child.first.data());
                const std::string second = boost::lexical_cast<std::string>(child.second.data());
				param.append_parameter(first, second);
            }
			params.push_back(param);
        } catch(...) {
            break;
        }
    }
	for(auto iter = params.begin(); iter != params.end(); ++iter) {
		iter->print_parameters();
	}
	
    return 0;
}

int main(int argc, char **argv)
{
    load_network_parameter_from_xml("logic.xml");
	return 0;
}

