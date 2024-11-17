using Microsoft.SemanticKernel;
using Azure.Identity;
using dotenv.net;

internal static class KernelFactory
{
    public static Kernel CreateKernel()
    {
        DotEnv.Load();

        string deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") 
            ?? throw new InvalidOperationException("Environment variable 'AZURE_OPENAI_DEPLOYMENT_NAME' is not set.");
        string endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") 
            ?? throw new InvalidOperationException("Environment variable 'AZURE_OPENAI_ENDPOINT' is not set.");
        
        return Kernel.CreateBuilder()
            .AddAzureOpenAIChatCompletion(deploymentName, endpoint, new DefaultAzureCredential())
            .Build();
    }
}