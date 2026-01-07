package dev.langchain4j.agentic.agent;

import static dev.langchain4j.agentic.declarative.DeclarativeUtil.*;
import static dev.langchain4j.agentic.internal.AgentUtil.*;
import static dev.langchain4j.internal.Utils.*;

import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

import dev.langchain4j.agentic.Agent;
import dev.langchain4j.agentic.internal.AgentUtil;
import dev.langchain4j.agentic.internal.AgenticScopeOwner;
import dev.langchain4j.agentic.internal.Context;
import dev.langchain4j.agentic.internal.InternalAgent;
import dev.langchain4j.agentic.internal.UserMessageRecorder;
import dev.langchain4j.agentic.observability.AgentListenerProvider;
import dev.langchain4j.agentic.scope.DefaultAgenticScope;
import dev.langchain4j.service.AiServiceContext;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.memory.ChatMemoryAccess;

public class AgentBuilder<T> extends BaseAgentBuilder<T, AgentBuilder<T>> {
    final Class<T> agentServiceClass;
    final Method agenticMethod;
    final Class<?> agentReturnType;

    public AgentBuilder(Class<T> agentServiceClass, Method agenticMethod) {
        this.agentServiceClass = agentServiceClass;
        this.agenticMethod = agenticMethod;
        this.agentReturnType = agenticMethod.getReturnType();

        Agent agent = agenticMethod.getAnnotation(Agent.class);
        if (agent == null) {
            throw new IllegalArgumentException("Method " + agenticMethod + " is not annotated with @Agent");
        }

        configureAgent(agentServiceClass, this);

        // Set the reflection-independent fields using setters from BaseAgentBuilder
        this.name(!isNullOrBlank(agent.name()) ? agent.name() : agenticMethod.getName());

        if (!isNullOrBlank(agent.description())) {
            this.description(agent.description());
        } else if (!isNullOrBlank(agent.value())) {
            this.description(agent.value());
        }

        this.outputKey(AgentUtil.outputKey(agent.outputKey(), agent.typedOutputKey()));

        this.async(agent.async());
        if (agent.summarizedContext() != null && agent.summarizedContext().length > 0) {
            this.summarizedContext(agent.summarizedContext());
        }
    }

    @Override
    protected Class<T> getAgentServiceClass() {
        return agentServiceClass;
    }

    @Override
    protected Class<?> getAgentReturnType() {
        return agentReturnType;
    }

    @Override
    T build(DefaultAgenticScope agenticScope) {
        this.arguments = argumentsFromMethod(agenticMethod, getDefaultValues());

        AiServiceContext context = AiServiceContext.create(agentServiceClass);
        AiServices<T> aiServices = AiServices.builder(context);
        if (getModel() != null) {
            aiServices.chatModel(getModel());
        }
        if (getChatMemory() != null) {
            aiServices.chatMemory(getChatMemory());
        }
        if (getChatMemoryProvider() != null) {
            aiServices.chatMemoryProvider(getChatMemoryProvider());
        }
        if (getSystemMessageProvider() != null) {
            aiServices.systemMessageProvider(getSystemMessageProvider());
        }
        if (getContentRetriever() != null) {
            aiServices.contentRetriever(getContentRetriever());
        }
        if (getRetrievalAugmentor() != null) {
            aiServices.retrievalAugmentor(getRetrievalAugmentor());
        }

        setupGuardrails(aiServices);
        setupTools(aiServices);

        UserMessageRecorder messageRecorder = new UserMessageRecorder();
        boolean agenticScopeDependent =
                getContextProvider() != null || (getContextProvidingAgents() != null && getContextProvidingAgents().length > 0);
        if (agenticScope != null && agenticScopeDependent) {
            if (getContextProvider() != null) {
                aiServices.chatRequestTransformer(
                        new Context.AgenticScopeContextGenerator(agenticScope, getContextProvider())
                                .andThen(messageRecorder));
            } else {
                aiServices.chatRequestTransformer(
                        new Context.Summarizer(agenticScope, getModel(), getContextProvidingAgents()).andThen(messageRecorder));
            }
        } else {
            aiServices.chatRequestTransformer(messageRecorder);
        }

        return (T) Proxy.newProxyInstance(
                agentServiceClass.getClassLoader(),
                new Class<?>[] {
                    agentServiceClass,
                    InternalAgent.class, AgentListenerProvider.class,
                    ChatMemoryAccess.class, AgenticScopeOwner.class,
                    ChatMessagesAccess.class
                },
                new AgentInvocationHandler(context, aiServices.build(), this, messageRecorder, agenticScopeDependent));
    }

    private void setupGuardrails(AiServices<T> aiServices) {
        if (getInputGuardrailsConfig() != null) {
            aiServices.inputGuardrailsConfig(getInputGuardrailsConfig());
        }
        if (getOutputGuardrailsConfig() != null) {
            aiServices.outputGuardrailsConfig(getOutputGuardrailsConfig());
        }
        if (getInputGuardrailClasses() != null) {
            aiServices.inputGuardrailClasses(getInputGuardrailClasses());
        }
        if (getOutputGuardrailClasses() != null) {
            aiServices.outputGuardrailClasses(getOutputGuardrailClasses());
        }
        if (getInputGuardrails() != null) {
            aiServices.inputGuardrails(getInputGuardrails());
        }
        if (getOutputGuardrails() != null) {
            aiServices.outputGuardrails(getOutputGuardrails());
        }
    }

    private void setupTools(AiServices<T> aiServices) {
        if (getObjectsWithTools() != null) {
            aiServices.tools(getObjectsWithTools());
        }
        if (getToolsMap() != null) {
            if (getImmediateReturnToolNames() != null) {
                aiServices.tools(getToolsMap(), getImmediateReturnToolNames());
            } else {
                aiServices.tools(getToolsMap());
            }
        }
        if (getToolProvider() != null) {
            aiServices.toolProvider(getToolProvider());
        }
        if (getMaxSequentialToolsInvocations() != null) {
            aiServices.maxSequentialToolsInvocations(getMaxSequentialToolsInvocations());
        }
        if (getHallucinatedToolNameStrategy() != null) {
            aiServices.hallucinatedToolNameStrategy(getHallucinatedToolNameStrategy());
        }
        if (isExecuteToolsConcurrently()) {
            if (getConcurrentToolsExecutor() != null) {
                aiServices.executeToolsConcurrently(getConcurrentToolsExecutor());
            } else {
                aiServices.executeToolsConcurrently();
            }
        }
        if (getToolArgumentsErrorHandler() != null) {
            aiServices.toolArgumentsErrorHandler(getToolArgumentsErrorHandler());
        }
        if (getToolExecutionErrorHandler() != null) {
            aiServices.toolExecutionErrorHandler(getToolExecutionErrorHandler());
        }
    }
}
