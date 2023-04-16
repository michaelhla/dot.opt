// basic c++ initialization

#include <iostream>
#include <memory>
#include <string>
using namespace std;

#include <grpcpp/grpcpp.h> // replace these lines with the generated stubs
#include "Dot.grpc.pb.h"   // still need to build

using Dot::Matrix;
using Dot::Operator;
using Dot::Res;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

class HandleOps final : public Operator::Service
{
    Status Multiply(ServerContext *context, const Matrix *request,
                    Res *reply) override
    {
        int m1 = request->rows();
        int n1 = request->cols();
        int m2 = n1;
        int n2 = 2;
        int m3 = m1;
        int n3 = n2;
        response->set_rows(m3);
        response->set_cols(n3);
        for (int i = 0; i < m3 * n3; i++)
        {
            response->add_data(0.0);
        }
        for (int i = 0; i < m3; i++)
        {
            for (int j = 0; j < n3; j++)
            {
                for (int k = 0; k < n1; k++)
                {
                    response->set_data(i * n3 + j, response->data(i * n3 + j) + request->data(i * n1 + k) * request->data(k * n2 + j));
                }
            }
        }
        return Status::OK;
    }
};

void dot(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

// server thread for primary backup
void server()
{
    // listen for requests
    // if request is for primary, send primary
    // if request is for backup, send backup
}

int main()
{
    std::string server_address("0.0.0.0:50051");
    MatrixServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();

    return 0;
}

// OUTLINE FOR DISTRIBUTED SYSTEMS PROJECT:
// 1. implement primary backup for N machines
// 1.a implement fast matrix multiplication/dot product algorithm
// 2. test dot product operations on single machine with CPU, GPU, then multiple machines with CPU
// 5. test training time of neural network on single machine with CPU, GPU, then multiple machines with CPU