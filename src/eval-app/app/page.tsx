import Image from "next/image";

import Link from "next/link"

export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <div>
          <h1 className="text-2xl font-bold">Basher evaluation app</h1>
          <p className="text-sm text-muted-foreground">This is a evaluation app for the Basher project. It is used to evaluate the performance of the Basher project.</p>
          <hr className="my-6 border-0 border-t border-border" />
          <p className="mt-5"><span className="font-bold">Evaluation Results:</span> Explore evaluation results and set scores for the evaluation results.</p>
          <p className="mt-5"><span className="font-bold">Run Evaluation:</span> Run a new evaluation.</p>
          <p className="mt-5"><span className="font-bold">Chat:</span> Chat with the Basher project. Plan evaluation tasks and export tasks to evaluation dataset.</p>
          <p className="mt-5"><span className="font-bold">Performance Monitor:</span> Monitor the overtime performance of the Basher project.</p>
          <p className="mt-5"><span className="font-bold">Qdrant Dashboard:</span> View the Qdrant dashboard.</p>
          <p className="mt-5"><span className="font-bold">API docs:</span> View the API docs.</p>
        </div>

      </main>
    </div>
  );
}
