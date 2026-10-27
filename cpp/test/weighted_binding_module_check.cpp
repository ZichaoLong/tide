#include "tide/weighted.h"
#include <dlfcn.h>
#include <iostream>
extern "C" void tide_test_weighted_install(tide::Model*,int);
extern "C" void tide_test_weighted_inspect(const tide::Model*,int);
int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("expected independent module path");
    void* module=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!module)throw std::runtime_error(dlerror());
    auto install=reinterpret_cast<void(*)(tide::Model*,int)>(dlsym(module,"tide_test_weighted_install"));
    auto inspect=reinterpret_cast<void(*)(const tide::Model*,int)>(dlsym(module,"tide_test_weighted_inspect"));
    if(!install||!inspect)throw std::runtime_error("missing independent binding interface");
    for(int family=0;family<5;++family){
      tide::Model foreign,local;install(&foreign,family);tide_test_weighted_inspect(&foreign,family);
      tide_test_weighted_install(&local,family);inspect(&local,family);
    }
    // Models must release their vtables before unloading the module.
    dlclose(module);std::cout<<"weighted cross-module inspection: five families, both directions passed\n";return 0;
  } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
